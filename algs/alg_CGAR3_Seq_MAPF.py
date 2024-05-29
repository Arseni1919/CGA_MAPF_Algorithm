from algs.alg_CGAR3_Seq_MAPF_functions import *


class AlgCgarMapf(AlgGeneric):

    name = 'CGAR-MAPF'

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
        self.agents: Deque[AlgCgar3SeqMapfAgent] = []
        self.agents_deq: Deque[AlgCgar3SeqMapfAgent] = deque()
        self.agents_dict: Dict[str, AlgCgar3SeqMapfAgent] = {}
        self.agents_num_dict: Dict[int, AlgCgar3SeqMapfAgent] = {}
        self.need_to_freeze_main_goal_node: bool = True
        self.n_agents = 0

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
        return [a.get_goal_node() for a in self.agents]

    @property
    def goal_nodes_names(self):
        return [a.get_goal_node().xy_name for a in self.agents]

    @property
    def n_solved(self) -> int:
        solved: List[AlgCgar3SeqMapfAgent] = [a for a in self.agents if a.goal_node == a.curr_node]
        return len(solved)

    def initialize_problem(self, obs: Dict[str, Any]) -> None:
        # create agents
        self.agents = deque()
        self.agents_dict = {}
        self.agents_num_dict = {}
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgCgar3SeqMapfAgent(num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes,
                                             nodes_dict=self.nodes_dict)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent
        self.n_agents = len(self.agents)
        self.agents_deq = deque(self.agents)

    def check_solvability(self) -> Tuple[bool, str]:
        # frankly, we need to check here the minimum number of free non-SV locations
        return True, 'good'

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[bool, Dict[str, List[Node]]]:
        """
        """

        # to render
        start_time = time.time()

        global_iteration = 0  # all at their start locations
        while not self.stop_condition():

            self.agents, last_n_name_to_a_dict, last_n_name_to_a_list = order_the_agents(self.agents)

            main_agent = self.agents[0]
            assert main_agent.priority == 0

            to_resume = cgar_conditions_check(
                main_agent, last_n_name_to_a_dict, last_n_name_to_a_list,
                self.non_sv_nodes_with_blocked_np, self.h_dict
            )
            if not to_resume:
                continue

            cgar_message, global_iteration = run_cgar(
                main_agent, self.agents, self.agents_dict, self.non_sv_nodes_with_blocked_np, self.h_dict, self.img_np,
                start_time, to_render, to_assert, self.name, self.n_solved, self.n_agents, global_iteration=global_iteration
            )

            # PRINT STATUS
            runtime = time.time() - start_time
            print(f'\r{'*' * 20} | [{self.name}] | i: {global_iteration} | {main_agent.name} | solved: {self.n_solved}/{self.n_agents} | runtime: {runtime: .2f} seconds | {'*' * 20}', end='')

            if cgar_message == 'start':
                continue
            elif cgar_message == 'end':
                self.agents.remove(main_agent)
                self.agents.append(main_agent)
                continue
            elif cgar_message == 'out':
                continue
            else:
                raise RuntimeError('nope')

        if to_assert:
            print(f'\n')
            for i in range(len(self.agents[0].path)):
                check_vc_ec_neic_iter(self.agents, i)
                print(f"checked {i}'th iteration")
            print('\n ------------- Paths are good! -------------')
        solved = self.stop_condition()
        paths_dict = compress_paths(self.agents, self.agents_dict, self.nodes, self.nodes_dict)
        for a in self.agents:
            a.path = paths_dict[a.name]
        # check_paths(self.agents)
        # paths_dict = {a.name: a.path for a in self.agents}
        return solved, paths_dict

    def stop_condition(self):
        for agent in self.agents:
            if agent.path[-1] != agent.goal_node:
                return False
        return True


@use_profiler(save_dir='../stats/alg_cgar_mapf.pstat')
def main():
    # single_mapf_run(AlgCgaMapf, is_SACGR=True)
    single_mapf_run(AlgCgarMapf, is_SACGR=False)


if __name__ == '__main__':
    main()

