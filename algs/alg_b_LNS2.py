import random

import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_temporal_a_star import calc_temporal_a_star, init_constraints, update_constraints


class AlgLNS2Agent:
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


def align_all_paths(agents: List[AlgLNS2Agent]) -> int:
    max_len = max([len(a.path) for a in agents])
    for a in agents:
        while len(a.path) < max_len:
            a.path.append(a.path[-1])
    return max_len


class AlgLNS2Mapf(AlgGeneric):

    name = 'LNS2'

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
        self.agents: List[AlgLNS2Agent] = []
        self.agents_dict: Dict[str, AlgLNS2Agent] = {}
        self.agents_num_dict: Dict[int, AlgLNS2Agent] = {}
        self.n_agents = 0
        self.max_path_len = 1000
        self.big_N = 5
        self.conf_matrix: np.ndarray | None = None
        self.conf_agents_names_list: List[str] = []
        self.conf_vv_random_walk: List[AlgLNS2Agent] | None = None
        self.conf_neighbourhood: List[AlgLNS2Agent] | None = None
        self.emp_vc_np = np.zeros((self.map_dim[0], self.map_dim[1], self.max_path_len))
        self.emp_ec_np = np.zeros((self.map_dim[0], self.map_dim[1], self.map_dim[0], self.map_dim[1], self.max_path_len))
        self.emp_pc_np = np.ones((self.map_dim[0], self.map_dim[1])) * -1

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
        solved: List[AlgLNS2Agent] = [a for a in self.agents if a.goal_node == a.curr_node]
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
            new_agent = AlgLNS2Agent(num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes,
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
        LNS2 - Collision-Based Neighborhoods:
        - build PrP
        - While there are collisions:
            - Build G_c
            - v <- select a random vertex v from Vc with deg(v) > 0
            - V_v <- find the connected component that has v
            - If |V_v| ≤ N:
                - put all inside
                - select random agent from V_v and do the random walk until fulfill V_v up to N
            - Otherwise:
                - Do the random walk from v in G_c until you have N agents
            - Solve the neighbourhood V_v with PrP (random priority)
            - Replace old plans with the new plans iff the number of colliding pairs (CP) of the paths in the new plan
              is no larger than that of the old plan
        return: valid plans
        """

        # to render
        start_time = time.time()
        iteration = 0
        # if to_render:
        #     plt.close()
        #     fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        #     plot_rate = 0.001
        #     # plot_rate = 2'

        self.build_initial_prp_plan(start_time, iteration, self.img_np)

        while (num_of_confs := self._build_G_c(start_time, iteration)) > 0:
            iteration += 1
            V_v, v = self._select_random_conf_v()
            if len(V_v) >= self.big_N:
                self.conf_neighbourhood = self.conf_vv_random_walk
            else:
                self._fill_the_neighbourhood(V_v)

            self._solve_with_PrP(start_time, iteration)
            # self._replace_old_plans()
            # print(f'\r{num_of_confs=}', end='')
            self.my_print(start_time, iteration, '', add_text=f'{num_of_confs=}')

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

    def my_print(self, start_time, iteration, solved, add_text=''):
        runtime = time.time() - start_time
        print(f'\r{'*' * 20} | [{self.name}] {iteration=} | solved: {solved}/{self.n_agents} |'
              f'runtime: {runtime: .2f} seconds | {add_text} | {'*' * 20}', end='')

    def build_initial_prp_plan(self, start_time, iteration, img_np: np.ndarray):
        vc_np, ec_np, pc_np = init_constraints(self.map_dim, max_path_len=self.max_path_len)
        max_final_time = 0

        for i, agent in enumerate(self.agents):
            new_path, info = calc_temporal_a_star(
                agent.curr_node, agent.goal_node, self.nodes_dict, self.h_dict, self.max_path_len,
                vc_np, ec_np, pc_np, max_final_time
            )
            # new_path, info = calc_temporal_a_star(
            #     agent.curr_node, agent.goal_node, self.nodes_dict, self.h_dict, self.max_path_len,
            #     self.emp_vc_np, self.emp_ec_np, self.emp_pc_np, max_final_time
            # )
            if new_path[-1] != agent.goal_node:
                new_path, info = calc_temporal_a_star(
                agent.curr_node, agent.goal_node, self.nodes_dict, self.h_dict, self.max_path_len,
                self.emp_vc_np, self.emp_ec_np, self.emp_pc_np, max_final_time
                )
                assert new_path[-1] == agent.goal_node

            vc_np, ec_np, pc_np = update_constraints(new_path, vc_np, ec_np, pc_np)
            max_final_time = max(max_final_time, len(new_path))
            agent.path = new_path
            # agent.build_v_e_maps(img_np)

            self.my_print(start_time, iteration, i, add_text='build_initial_prp_plan')

    def _build_G_c(self, start_time, iteration):
        num_of_agents = self.n_agents
        self.conf_matrix = np.zeros((num_of_agents, num_of_agents))
        self.conf_agents_names_list = []
        num_of_confs = 0
        for i_comp, (agent1, agent2) in enumerate(combinations(self.agents, 2)):
            if not two_plans_have_no_confs(agent1.path, agent2.path):
                num_of_confs += 1
                self.conf_matrix[agent1.num, agent2.num] = 1
                self.conf_matrix[agent2.num, agent1.num] = 1
                self.conf_agents_names_list.append(agent1.name)
                self.conf_agents_names_list.append(agent2.name)
            self.my_print(start_time, iteration, i_comp, add_text=f'_build_G_c | {num_of_confs=}')
        self.conf_agents_names_list = list(set(self.conf_agents_names_list))
        return num_of_confs

    def _select_random_conf_v(self):
        self.conf_vv_random_walk = []
        v = self.agents_dict[random.choice(self.conf_agents_names_list)]
        V_v, V_v_nums = [], []
        new_leaves = [v.num]
        while len(new_leaves) > 0:
            next_leave = new_leaves.pop(0)
            if len(self.conf_vv_random_walk) < self.big_N and next_leave != v.num:
                self.conf_vv_random_walk.append(self.agents_dict[f'agent_{next_leave}'])
                V_v_nums.append(next_leave)
            elif len(self.conf_vv_random_walk) == self.big_N:
                break
            children = np.where(self.conf_matrix[next_leave] == 1)[0]
            new_leaves.extend([c for c in children if c not in V_v_nums])
            new_leaves = list(set(new_leaves))
        V_v = [self.agents_dict[f'agent_{num}'] for num in V_v_nums]
        return V_v, v

    def _fill_the_neighbourhood(self, V_v):
        self.conf_neighbourhood = V_v
        not_in_nbhd = [agent for agent in self.agents if agent not in self.conf_neighbourhood]
        while len(self.conf_neighbourhood) < self.big_N:
            new_one = random.choice(not_in_nbhd)
            self.conf_neighbourhood.append(new_one)

    def _solve_with_PrP(self, start_time, iteration):
        print()
        # reset
        vc_np, ec_np, pc_np = init_constraints(self.map_dim, max_path_len=self.max_path_len)
        max_final_time = 0
        h_agents = [agent for agent in self.agents if agent not in self.conf_neighbourhood]
        for h_agent in h_agents:
            new_path = h_agent.path
            vc_np, ec_np, pc_np = update_constraints(new_path, vc_np, ec_np, pc_np)
            max_final_time = max(max_final_time, len(new_path))
        random.shuffle(self.conf_neighbourhood)

        # for agent in self.conf_neighbourhood:
        #     agent.path = []

        for i, agent in enumerate(self.conf_neighbourhood):
            new_path, info = calc_temporal_a_star(
                agent.curr_node, agent.goal_node, self.nodes_dict, self.h_dict, self.max_path_len,
                vc_np, ec_np, pc_np, max_final_time
            )
            if new_path[-1] != agent.goal_node:
                new_path, info = calc_temporal_a_star(
                    agent.curr_node, agent.goal_node, self.nodes_dict, self.h_dict, self.max_path_len,
                    self.emp_vc_np, self.emp_ec_np, self.emp_pc_np, max_final_time
                )
            vc_np, ec_np, pc_np = update_constraints(new_path, vc_np, ec_np, pc_np)
            max_final_time = max(max_final_time, len(new_path))
            agent.path = new_path

            self.my_print(start_time, iteration, i, add_text='_solve_with_PrP')


@use_profiler(save_dir='../stats/alg_lns2_mapf.pstat')
def main():
    # single_mapf_run(AlgLNS2Mapf, is_SACGR=True)
    single_mapf_run(AlgLNS2Mapf, is_SACGR=False)


if __name__ == '__main__':
    main()




