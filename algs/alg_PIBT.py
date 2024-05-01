import heapq
from abc import ABC

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric


def build_vc_ec_from_configs(config_from: Dict[str, Node], config_to: Dict[str, Node]):
    vc_set, ec_set = [], []
    for agent_name, node_from in config_from.items():
        if agent_name in config_to:
            node_to = config_to[agent_name]
            heapq.heappush(vc_set, (node_to.x, node_to.y))
            heapq.heappush(ec_set, (node_from.x, node_from.y, node_to.x, node_to.y))
    return vc_set, ec_set


def procedure_i_pibt[T](
        agent: T,
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals: Dict[str, Node],
        node_name_to_agent_dict: Dict[str, T],
        node_name_to_agent_list: List[str],
) -> bool:
    agent_name = agent.name
    agent_curr_node = config_from[agent_name]
    h_goal_np: np.ndarray = h_dict[goals[agent_name].xy_name]
    vc_set, ec_set = build_vc_ec_from_configs(config_from, config_to)

    # sort C in ascending order of dist(u, gi) where u ∈ C
    nei_nodes: List[Node] = [nodes_dict[n_name] for n_name in config_from[agent_name].neighbours]
    random.shuffle(nei_nodes)

    def get_nei_v(n: Node):
        # s_v = 0.5 if n.xy_name in node_name_to_agent_list else 0
        s_v = 0
        return h_goal_np[n.x, n.y] + s_v

    nei_nodes.sort(key=get_nei_v)

    for j, nei_node in enumerate(nei_nodes):
        # vc
        if (nei_node.x, nei_node.y) in vc_set:
            continue
        # ec
        if (nei_node.x, nei_node.y, agent_curr_node.x, agent_curr_node.y) in ec_set:
            continue

        config_to[agent_name] = nei_node
        if nei_node.xy_name in node_name_to_agent_list:
            next_agent = node_name_to_agent_dict[nei_node.xy_name]
            if agent != next_agent and next_agent.name not in config_to:
                next_is_valid = procedure_i_pibt(
                    next_agent, nodes_dict, h_dict, config_from, config_to, goals,
                    node_name_to_agent_dict, node_name_to_agent_list,
                )
                if not next_is_valid:
                    continue
        return True
    config_to[agent_name] = agent_curr_node
    return False


def run_i_pibt[T](
        main_agent: T,
        agents: List[T],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        config_from: Dict[str, Node] | None = None,
        config_to: Dict[str, Node] | None = None,
        goals: Dict[str, Node] | None = None,
        node_name_to_agent_dict: Dict[str, T] | None = None,
        node_name_to_agent_list: List[str] | None = None,
) -> Dict[str, Node]:
    if config_from is None:
        config_from: Dict[str, Node] = {agent.name: agent.curr_node for agent in agents}
    if config_to is None:
        config_to: Dict[str, Node] = {}
    if goals is None:
        goals: Dict[str, Node] = {agent.name: agent.goal_node for agent in agents}
    if node_name_to_agent_dict is None:
        node_name_to_agent_dict: Dict[str, T] = {a.curr_node.xy_name: a for a in agents}
        node_name_to_agent_list: List[str] = list(node_name_to_agent_dict.keys())
        heapq.heapify(node_name_to_agent_list)
    _ = procedure_i_pibt(main_agent, nodes_dict, h_dict, config_from, config_to, goals,
                         node_name_to_agent_dict, node_name_to_agent_list)
    return config_to


def run_pibt[T](agents: List[T], nodes_dict: Dict[str, Node], h_dict: Dict[str, np.ndarray],
                iteration: int) -> Dict[str, Node]:
    config_from: Dict[str, Node] = {agent.name: agent.curr_node for agent in agents}
    goals: Dict[str, Node] = {agent.name: agent.goal_node for agent in agents}
    config_to: Dict[str, Node] = {}
    node_name_to_agent_dict: Dict[str, T] = {a.curr_node.xy_name: a for a in agents}
    node_name_to_agent_list: List[str] = list(node_name_to_agent_dict.keys())
    heapq.heapify(node_name_to_agent_list)
    for agent in agents:
        # already planned
        if agent.name in config_to:
            continue
        # no target
        if agent.curr_node == agent.goal_node:
            continue
        if agent.name not in config_to:
            valid = run_i_pibt(agent, agents, nodes_dict, h_dict, config_from, config_to, goals,
                               node_name_to_agent_dict, node_name_to_agent_list)
    for agent in agents:
        if agent.name not in config_to:
            config_to[agent.name] = agent.curr_node
    return config_to


class AlgPIBTAgent:
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


class AlgPIBT(AlgGeneric):
    name = 'PIBT'

    def __init__(self, env: SimEnvMAPF):
        super().__init__()
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
        self.agents: List[AlgPIBTAgent] = []
        self.agents_dict: Dict[str, AlgPIBTAgent] = {}
        self.main_agent: AlgPIBTAgent | None = None

        # logs
        self.logs: dict | None = None

    @property
    def n_agents(self):
        return len(self.agents)

    def initialize_problem(self, obs: Dict[str, Any]) -> None:
        # create agents
        self.agents: List[AlgPIBTAgent] = []
        self.agents_dict: Dict[str, AlgPIBTAgent] = {}
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgPIBTAgent(num=num, start_node=start_node, goal_node=goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def check_solvability(self) -> Tuple[bool, str]:
        return True, 'good'

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[
        bool, Dict[str, List[Node]]]:

        # to render
        if to_render:
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            plot_rate = 0.001

        finished = False
        iteration = 0
        while not finished:

            # calc the step
            config_to = run_pibt(self.agents, self.nodes_dict, self.h_dict, iteration)

            # execute the step + check the termination condition
            finished = True
            agents_finished, agents_unfinished = [], []
            for agent in self.agents:
                next_node = config_to[agent.name]
                agent.path.append(next_node)
                agent.prev_node = agent.curr_node
                agent.curr_node = next_node
                if agent.curr_node != agent.goal_node:
                    finished = False
                    agents_unfinished.append(agent)
                else:
                    agents_finished.append(agent)

            # unfinished first
            # random.shuffle(agents_finished)
            # random.shuffle(agents_unfinished)
            # agents_unfinished = agents_unfinished[1:] + agents_unfinished[:1]
            self.agents: List[AlgPIBTAgent] = agents_unfinished
            self.agents.extend(agents_finished)

            # stats

            # print + render
            print(f'\r[PIBT] {iteration=} | finished: {len(agents_finished)}/{self.n_agents}', end='')
            iteration += 1

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

        paths_dict = {a.name: a.path for a in self.agents}
        return True, paths_dict


@use_profiler(save_dir='../stats/alg_pibt.pstat')
def main():
    single_mapf_run(AlgPIBT, is_SACGR=False)
    # single_mapf_run(AlgCGAR, is_SACGR=True)


if __name__ == '__main__':
    main()
