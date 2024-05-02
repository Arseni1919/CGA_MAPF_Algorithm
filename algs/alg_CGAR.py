from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt


class AlgCGARAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_node: Node = start_node
        self.goal_node: Node = goal_node
        self.path: List[Node] = [start_node]
        self.arrived: bool = False

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def last_path_node_name(self):
        return self.path[-1].xy_name

    @property
    def a_curr_node_name(self):
        return self.curr_node.xy_name

    @property
    def a_prev_node_name(self):
        return self.prev_node.xy_name

    @property
    def a_next_node_name(self):
        return self.next_node.xy_name

    @property
    def a_goal_node_name(self):
        return self.goal_node.xy_name

    def __eq__(self, other):
        return self.num == other.num


class AlgCGAR(AlgGeneric):
    name = 'CGAR'

    def __init__(self, env: SimEnvMAPF):
        super().__init__()
        assert env.is_SACGR

        self.env = env

        # for the map
        self.img_dir = self.env.img_dir
        self.nodes, self.nodes_dict = copy_nodes(self.env.nodes)
        self.img_np: np.ndarray = self.env.img_np
        self.non_sv_nodes_np: np.ndarray = self.env.non_sv_nodes_np
        self.map_dim = self.env.map_dim
        self.h_func = self.env.h_func
        # the structure of h_dict: h_dict[xy_name] -> np.ndarray
        self.h_dict = self.env.h_dict

        # for the problem
        self.agents: List[AlgCGARAgent] = []
        self.agents_dict: Dict[str, AlgCGARAgent] = {}
        self.main_agent: AlgCGARAgent | None = None

        # logs
        self.logs: dict | None = None

    @property
    def start_nodes(self):
        return [a.start_node for a in self.agents]

    @property
    def start_nodes_names(self):
        return [a.start_node.xy_name for a in self.agents]

    @property
    def goal_nodes(self):
        return [a.goal_node for a in self.agents]

    @property
    def goal_nodes_names(self):
        return [a.goal_node.xy_name for a in self.agents]

    def initialize_problem(self, obs: Dict[str, Any]) -> None:
        # create agents
        self.agents = []
        self.agents_dict = {}
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgCGARAgent(num=num, start_node=start_node, goal_node=goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

        self.main_agent = self.agents_dict[obs['main_agent_name']]

    def check_solvability(self) -> Tuple[bool, str]:
        assert self.main_agent.goal_node != self.main_agent.start_node
        # if somebody stands in the goal location of the main agent
        if self.main_agent.goal_node in self.start_nodes:
            return False, 'goal_occupied'
        # if the current location is a non-SV
        if self.non_sv_nodes_np[self.main_agent.curr_node.x, self.main_agent.curr_node.y]:
            return True, 'good'
        # if there are enough free locations to allow the main agent to pass
        # TODO: check the function ↓
        return is_enough_free_locations(self.main_agent.start_node, self.main_agent.goal_node, self.nodes_dict,
                                        self.h_dict, self.start_nodes, self.non_sv_nodes_np)

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[bool, Dict[str, List[Node]]]:
        """
        - Block the goal vertex
        - While curr_node is not the goal:
            - If the plan is needed:
                - If the next vertex is non-SV:
                    - build PIBT step
                - Else:
                    - Build corridor
                    - Build EP for ev-agents in the corridor
                    - Evacuate ev-agents
                    - Build the steps in the corridor to the main agent
            - execute step
        - Reverse all agents that where moved away -> return
        """
        # to render
        if to_render:
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            plot_rate = 0.001

        iteration = 0
        while self.main_agent.curr_node != self.main_agent.goal_node:
            if len(self.main_agent.path) - 1 < iteration:

                # assert
                if to_assert:
                    for agent in self.agents:
                        assert len(agent.path) - 1 < iteration

                # if you are here, there is a need for a plan for a future step
                main_next_node = get_min_h_nei_node(self.main_agent.curr_node, self.main_agent.goal_node, self.nodes_dict, self.h_dict)
                if self.non_sv_nodes_np[main_next_node.x, main_next_node.y]:
                    # calc single PIBT step
                    self.calc_pibt_step(iteration)
                else:
                    # calc evacuation of agents from the corridor
                    self.calc_ep_steps(iteration)

            # execute the step
            for agent in self.agents:
                next_node = agent.path[iteration]
                agent.prev_node = agent.curr_node
                agent.curr_node = next_node

            # updates after the step execution
            iteration += 1

            # print + render
            print(f'\r[CGAR] {iteration=} | ', end='')

            if to_render and iteration >= 0:
                # i_agent = self.agents_dict['agent_0']
                i_agent = self.agents[0]
                plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': i_agent, }
                plot_step_in_env(ax[0], plot_info)
                # plot_total_finished_goals(ax[1], plot_info)
                # plot_unique_movements(ax[1], plot_info)
                plt.pause(plot_rate)
                if i_agent.prev_node == i_agent.curr_node:
                    print(f'\n{iteration=} | {i_agent.name=}')

        # reverse part
        pass

        paths_dict = {a.name: a.path for a in self.agents}
        return True, paths_dict

    def calc_pibt_step(self, iteration: int):
        assert len(self.main_agent.path) - 1 < iteration
        # preps
        config_to = {}
        for agent in self.agents:
            if len(agent.path) - 1 >= iteration:
                config_to[agent.name] = agent.path[iteration]

        # calc PIBT
        config_to = run_i_pibt(self.main_agent, self.agents, self.nodes_dict, self.h_dict, config_to=config_to)
        for agent in self.agents:
            if agent.name not in config_to:
                config_to[agent.name] = agent.curr_node

        # extend the paths
        for agent in self.agents:
            next_node = config_to[agent.name]
            agent.path.append(next_node)

    def calc_ep_steps(self, iteration: int):
        """
        - Build corridor
        - Build EP for ev-agents in the corridor
        - Evacuate ev-agents
        - Build the steps in the corridor to the main agent
        """
        # Build corridor
        pass

        # Build EP for ev-agents in the corridor
        pass

        # Evacuate ev-agents
        pass

        # Build the steps in the corridor to the main agent
        pass

        # Extend the paths
        pass


@use_profiler(save_dir='../stats/alg_cgar.pstat')
def main():
    single_mapf_run(AlgCGAR, is_SACGR=True)


if __name__ == '__main__':
    main()
