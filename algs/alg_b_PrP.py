import random

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_temporal_a_star import calc_temporal_a_star, init_constraints, update_constraints


class AlgPrPAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node, nodes: List[Node], nodes_dict: Dict[str, Node]):
        self.num = num
        self.priority = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.path: List[Node] = [start_node]

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
    def a_goal_node_name(self):
        return self.get_goal_node().xy_name

    @property
    def a_start_node_name(self):
        return self.start_node.xy_name

    @property
    def is_moving(self):
        return self.curr_node != self.get_goal_node()

    @property
    def temp_arrived(self):
        return self.curr_node == self.get_goal_node()

    def __eq__(self, other):
        return self.num == other.num

    def __lt__(self, other):
        return self.num < other.num

    def __gt__(self, other):
        return self.num > other.num

    def __hash__(self):
        return hash(self.num)

    def get_goal_node(self) -> Node:
        return self.goal_node


def align_all_paths(agents: List[AlgPrPAgent]) -> int:
    max_len = max([len(a.path) for a in agents])
    for a in agents:
        while len(a.path) < max_len:
            a.path.append(a.path[-1])
    return max_len


class AlgPrPMapf(AlgGeneric):

    name = 'PrP'

    def __init__(self, env: SimEnvMAPF):
        super().__init__()

        self.env = env

        # for the map
        self.img_dir: str = self.env.img_dir
        self.nodes, self.nodes_dict = copy_nodes(self.env.nodes)
        self.img_np: np.ndarray = self.env.img_np
        # self.non_sv_nodes_np: np.ndarray = self.env.non_sv_nodes_np
        self.non_sv_nodes_with_blocked_np: np.ndarray = self.env.non_sv_nodes_with_blocked_np
        self.map_dim: Tuple[int, int] = self.env.map_dim
        self.h_func = self.env.h_func
        # the structure of h_dict: h_dict[xy_name] -> np.ndarray
        self.h_dict: Dict[str, np.ndarray] = self.env.h_dict

        # for the problem
        self.agents: List[AlgPrPAgent] = []
        self.agents_dict: Dict[str, AlgPrPAgent] = {}
        self.agents_num_dict: Dict[int, AlgPrPAgent] = {}
        self.n_agents = 0
        self.max_path_len = 1000

        # logs
        self.logs: dict | None = None

    @property
    def start_nodes(self):
        return [a.start_node for a in self.agents]

    @property
    def start_nodes_names(self):
        return [a.start_node.xy_name for a in self.agents]

    @property
    def curr_nodes(self):
        return [a.curr_node for a in self.agents]

    @property
    def curr_nodes_names(self):
        return [a.curr_node.xy_name for a in self.agents]

    @property
    def goal_nodes(self):
        return [a.get_goal_node() for a in self.agents]

    @property
    def goal_nodes_names(self):
        return [a.get_goal_node().xy_name for a in self.agents]

    @property
    def n_solved(self) -> int:
        solved: List[AlgPrPAgent] = [a for a in self.agents if a.goal_node == a.curr_node]
        return len(solved)

    def initialize_problem(self, obs: Dict[str, Any]) -> None:
        # create agents
        self.agents = []
        self.agents_dict = {}
        self.agents_num_dict = {}
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgPrPAgent(num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes,
                                    nodes_dict=self.nodes_dict)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent
        self.n_agents = len(self.agents)

    def check_solvability(self) -> Tuple[bool, str]:
        # frankly, we need to check here the minimum number of free non-SV locations
        return True, 'good'

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[bool, Dict[str, List[Node]]]:
        """
        """

        # to render
        start_time = time.time()
        # if to_render:
        #     plt.close()
        #     fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        #     plot_rate = 0.001
        #     # plot_rate = 2'

        iteration = 0  # all at their start locations
        while not self.stop_condition():
            # current step iteration
            iteration += 1

            # update priorities
            self.update_priorities()

            # calc paths
            succeeded = self.calc_paths(start_time, iteration)

            # print + render
            # runtime = time.time() - start_time
            # print(f'\r{'*' * 20} | [{self.name}] {iteration=} | solved: {self.n_solved}/{self.n_agents} |'
            #       f'runtime: {runtime: .2f} seconds | {'*' * 20}', end='')
            # if to_render and iteration >= 0:
            #     i_agent = self.agents[0]
            #     plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': i_agent}
            #     plot_step_in_env(ax[0], plot_info)
            #     plt.pause(plot_rate)

        if to_assert:
            print(f'\n')
            min_len = min([len(a.path) for a in self.agents])
            for i in range(min_len):
                check_vc_ec_neic_iter(self.agents, i)
                print(f"checked {i}'th iteration")
            print('\n ------------- Paths are good! -------------')
        align_all_paths(self.agents)
        paths_dict = {a.name: a.path for a in self.agents}
        solved = self.stop_condition()
        return solved, paths_dict

    def stop_condition(self):
        for agent in self.agents:
            if len(agent.path) == 0:
                return False
            if agent.path[-1] != agent.goal_node:
                return False
        return True

    def update_priorities(self):
        random.shuffle(self.agents)

    def calc_paths(self, start_time, iteration):
        vc_np, ec_np, pc_np = init_constraints(self.map_dim, max_path_len=self.max_path_len)
        max_final_time = 0

        for agent in self.agents:
            agent.path = []

        for i, agent in enumerate(self.agents):
            new_path, info = calc_temporal_a_star(
                agent.curr_node, agent.goal_node, self.nodes_dict, self.h_dict, self.max_path_len,
                vc_np, ec_np, pc_np, max_final_time
            )
            if new_path[-1] != agent.goal_node:
                return False
            vc_np, ec_np, pc_np = update_constraints(new_path, vc_np, ec_np, pc_np)
            max_final_time = max(max_final_time, len(new_path))
            agent.path = new_path

            runtime = time.time() - start_time
            print(f'\r{'*' * 20} | [{self.name}] {iteration=} | solved: {i}/{self.n_agents} |'
                  f'runtime: {runtime: .2f} seconds | {'*' * 20}', end='')
            # print()

        return True


@use_profiler(save_dir='../stats/alg_prp_mapf.pstat')
def main():
    # single_mapf_run(AlgLNS2Mapf, is_SACGR=True)
    single_mapf_run(AlgPrPMapf, is_SACGR=False)


if __name__ == '__main__':
    main()




