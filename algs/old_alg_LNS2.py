from typing import List, Dict

import numpy as np

from tools_for_graph_nodes import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.test_single_alg import test_single_alg
from algs.alg_a_star_space_time import a_star_xyt
from algs.alg_ParObs_PF_PrP_seq import ParObsPFPrPAgent, AlgParObsPFPrPSeq


class LNS2Agent(ParObsPFPrPAgent):
    def __init__(self, num: int, start_node, next_goal_node, **kwargs):
        super().__init__(num, start_node, next_goal_node, **kwargs)

    # POTENTIAL FIELDS ****************************** pf_weight ******************************
    def _build_nei_pfs(self, h_agents):
        if self.pf_weight == 0:
            self.nei_pfs = None
            return None, None
        if len(h_agents) == 0:
            return None, None
        max_plan_len = max([len(agent.plan) for agent in h_agents])
        nei_pfs = np.zeros((self.map_dim[0], self.map_dim[1], max_plan_len))  # x, y, t
        for nei_agent in h_agents:
            up_until_t = len(nei_agent.plan)
            weight = self._get_weight(nei_heuristic_value=nei_agent.heuristic_value)
            nei_pfs[:, :, :up_until_t] += weight * nei_agent.pf_field

        # # ---------- memory part ---------- # #
        # self.memory *= 0.9
        # norm_memory = np.repeat(self.memory[:, :, np.newaxis], max_plan_len, axis=2)
        # memory_weight = 3
        # norm_memory *= memory_weight
        # print(norm_memory)

        self.nei_pfs = nei_pfs
        # nei_pfs += norm_memory
        return nei_pfs, max_plan_len


class AlgLNS2Seq(AlgParObsPFPrPSeq):
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    def __init__(self, params, alg_name):
        super().__init__(params, alg_name)
        self.big_N = params['big_N']
        self.conf_matrix = None
        self.conf_agents_names_list = None
        self.conf_vv_random_walk = None
        self.conf_neighbourhood = None

    def reset(self):
        self.agents: List[ParObsPFPrPAgent] = []
        for env_agent in self.env.agents:
            new_agent = LNS2Agent(
                num=env_agent.num, start_node=env_agent.start_node, next_goal_node=env_agent.next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim, params=self.params
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.curr_iteration = 0
        self.i_agent = self.agents_dict['agent_0']

    def _build_G_c(self):
        num_of_agents = len(self.agents)
        self.conf_matrix = np.zeros((num_of_agents, num_of_agents))
        self.conf_agents_names_list = []
        num_of_confs = 0
        for agent1, agent2 in combinations(self.agents, 2):
            if agent1.name not in agent2.nei_dict:
                continue
            if not two_plans_have_no_confs(agent1.plan, agent2.plan):
                num_of_confs += 1
                self.conf_matrix[agent1.num, agent2.num] = 1
                self.conf_matrix[agent2.num, agent1.num] = 1
                self.conf_agents_names_list.append(agent1.name)
                self.conf_agents_names_list.append(agent2.name)
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

    def _solve_with_PrP(self):
        random.shuffle(self.conf_neighbourhood)
        # reset
        for agent in self.conf_neighbourhood:
            agent.plan = None

        h_agents = [agent for agent in self.agents if agent not in self.conf_neighbourhood]
        for agent in self.agents:
            agent.build_plan(h_agents)
            h_agents.append(agent)

    def _replace_old_plans(self):
        pass

    def _build_plans(self):
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return {}
        """
        LNS2:
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
        start_time = time.time()
        # self._update_order()
        self._reshuffle_agents()

        time_limit_crossed = self.initial_prp_assignment(start_time)
        if time_limit_crossed:
            return {}

        while (num_of_confs := self._build_G_c()) > 0:
            V_v, v = self._select_random_conf_v()
            if len(V_v) >= self.big_N:
                self.conf_neighbourhood = self.conf_vv_random_walk
            else:
                self._fill_the_neighbourhood(V_v)

            self._solve_with_PrP()
            self._replace_old_plans()
            # print(f'\n{num_of_confs=}')
            print(f'\r{num_of_confs=}', end='')

            # limit check
            end_time = time.time() - start_time
            if end_time > self.time_to_think_limit:
                self._implement_istay()
                return {}
        return {}


@use_profiler(save_dir='../stats/old_alg_lns2.pstat')
def main():
    # Alg params
    pf_weight = 0.5
    pf_size = 4
    big_N = 5
    h = 5
    w = h
    # alg_name = 'LNS2'
    # alg_name = 'PF-LNS2'
    alg_name = 'ParObs-LNS2'
    # alg_name = 'ParObs-PF-LNS2'

    params_dict = {
        'LNS2': {'big_N': big_N},
        'PF-LNS2': {
            'big_N': big_N,
            # For PF
            'pf_weight': 0.5,
            # 'pf_weight': 1,
            # 'pf_weight': 3,
            # 'pf_weight': 2,
            # 'pf_size': 'h',
            'pf_size': 3,
            # 'pf_size': 5,
            # 'pf_size': 2,
        },
        'ParObs-LNS2': {
            'big_N': big_N,
            # For RHCR
            'h': 5,  # my step
            'w': 5,  # my planning
        },
        'ParObs-PF-LNS2': {
            'big_N': big_N,
            'h': h, 'w': w, 'pf_weight': pf_weight, 'pf_size': pf_size
        },
    }

    alg = AlgLNS2Seq(params=params_dict[alg_name], alg_name=alg_name)
    test_single_alg(
        alg,

        # GENERAL
        # random_seed=True,
        random_seed=False,
        seed=321,
        PLOT_PER=1,
        # PLOT_PER=20,
        PLOT_RATE=0.001,
        # PLOT_RATE=5,
        PLOT_FROM=1,
        middle_plot=True,
        # middle_plot=False,
        final_plot=True,
        # final_plot=False,

        # FOR ENV
        # iterations=50,  # !!!
        # iterations=200,
        iterations=100,
        n_agents=100,
        n_problems=1,
        # classical_rhcr_mapf=True,
        classical_rhcr_mapf=False,
        time_to_think_limit=3000,  # seconds
        rhcr_mapf_limit=10000,
        global_time_limit=6000,  # seconds

        # Map
        # img_dir='10_10_my_rand.map',  # 10-10
        # img_dir='10_10_my_empty.map',  # 10-10

        # img_dir='empty-32-32.map',  # 32-32
        img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        # img_dir='random-32-32-20.map',  # 32-32
        # img_dir='room-32-32-4.map',  # 32-32
        # img_dir='maze-32-32-2.map',  # 32-32

        # img_dir='warehouse-10-20-10-2-1.map',
    )


if __name__ == '__main__':
    main()

