from algs.alg_CGAR2_MAPF_functions import *


class AlgCgar2Mapf(AlgGeneric):

    name = 'CGAR2-MAPF'

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
        self.agents: List[AlgCgar2MapfAgent] = []
        self.agents_dict: Dict[str, AlgCgar2MapfAgent] = {}
        self.agents_num_dict: Dict[int, AlgCgar2MapfAgent] = {}
        self.agents_to_return_dict: Dict[str, List[AlgCgar2MapfAgent]] = {}
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
        solved: List[AlgCgar2MapfAgent] = [a for a in self.agents if a.goal_node == a.curr_node]
        return len(solved)

    @property
    def agents_to_return_dict_names(self):
        agents_to_return_dict_names = {}
        for k, v in self.agents_to_return_dict.items():
            new_v = [a.name for a in v]
            agents_to_return_dict_names[k] = new_v
        return agents_to_return_dict_names

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
            new_agent = AlgCgar2MapfAgent(
                num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes, nodes_dict=self.nodes_dict
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent
            self.agents_to_return_dict[new_agent.name] = []
        self.n_agents = len(self.agents)

    def check_solvability(self) -> Tuple[bool, str]:
        # frankly, we need to check here the minimum number of free non-SV locations
        return True, 'good'

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[bool, Dict[str, List[Node]]]:
        """
        """

        # to render
        start_time = time.time()
        if to_render:
            plt.close()
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            plot_rate = 0.001
            # plot_rate = 2

        iteration = 0  # all at their start locations
        while not self.stop_condition():
            # next step iteration
            iteration += 1
            # UPDATE ORDER
            self.update_priorities(iteration, to_assert)
            # BUILD NEXT STEP
            self.build_next_steps(iteration, to_assert)
            # EXECUTE THE STEP
            self.execute_next_steps(iteration, to_assert)

            # PRINT
            runtime = time.time() - start_time
            print(f'\r{'*' * 20} | [{self.name}] {iteration=} | solved: {self.n_solved}/{self.n_agents} | runtime: {runtime: .2f} seconds | {'*' * 20}', end='')
            # RENDER
            if to_render and iteration >= 0:
                i_agent = self.agents[0]
                non_sv_nodes_np = self.non_sv_nodes_with_blocked_np[i_agent.get_goal_node().x, i_agent.get_goal_node().y]
                plot_info = {
                    'img_np': self.img_np,
                    'agents': self.agents,
                    'i_agent': i_agent,
                    'agents_to_return_dict': self.agents_to_return_dict,
                    'non_sv_nodes_np': non_sv_nodes_np
                }
                plot_step_in_env(ax[0], plot_info)
                plot_return_paths(ax[1], plot_info)
                plt.pause(plot_rate)

        if to_assert:
            print(f'\n')
            for i in range(len(self.agents[0].path)):
                check_vc_ec_neic_iter(self.agents, i)
                print(f"checked {i}'th iteration")
            print('\n ------------- Paths are good! -------------')
        # TODO: cut all paths to the minimum
        paths_dict = {a.name: a.path for a in self.agents}
        solved = self.stop_condition()
        return solved, paths_dict

    def stop_condition(self):
        for agent in self.agents:
            if agent.path[-1] != agent.goal_node:
                return False
        return True

    def update_priorities(self, iteration: int, to_assert: bool = False) -> None:
        self.agents.sort(key=lambda a: a.future_rank)
        unfinished: List[AlgCgar2MapfAgent] = []
        finished: List[AlgCgar2MapfAgent] = []
        for agent in self.agents:
            return_agents = self.agents_to_return_dict[agent.name]
            if agent.curr_node == agent.get_goal_node() and len(return_agents) == 0:
                finished.append(agent)
            else:
                unfinished.append(agent)
        self.agents = [*unfinished, *finished]
        update_priority_numbers(self.agents)

    def build_next_steps(self, iteration: int, to_assert: bool = False) -> None:

        config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list = self.get_preparations(iteration)
        update_status(self.agents)
        future_captured_node_names: List[str] = []
        future_captured_node_names = update_future_captured_node_names(future_captured_node_names, self.agents, iteration)

        # update current rank
        for curr_rank, agent in enumerate(self.agents):
            agent.curr_rank = curr_rank

        for curr_rank, agent in enumerate(self.agents):
            hr_agents = self.agents[:curr_rank]
            lr_agents = self.agents[curr_rank + 1:]

            to_resume, cc_stage_info = continuation_check_stage(
                agent, hr_agents, lr_agents, iteration, curr_n_name_to_a_dict, curr_n_name_to_a_list, goals_dict,
                self.agents_to_return_dict, self.agents, self.agents_dict, self.img_np, self.h_dict,
                self.non_sv_nodes_with_blocked_np, self.nodes, self.nodes_dict,
            )
            if not to_resume:
                continue

            unplanned_agents: List[AlgCgar2MapfAgent] = [
                a for a in self.agents if len(a.path) - 1 == iteration - 1 and a != agent
            ]

            calc_step_stage(
                agent, hr_agents, lr_agents, iteration, config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                self.non_sv_nodes_with_blocked_np, self.agents, self.agents_dict, self.agents_to_return_dict, self.nodes, self.nodes_dict,
                self.img_np, self.h_dict
            )

            # Get newly-moved agents
            newly_planned_agents: List[AlgCgar2MapfAgent] = [
                a for a in unplanned_agents if len(a.path) - 1 >= iteration
            ]
            set_parent_of_path(newly_planned_agents, parent=agent)
            future_captured_node_names = update_future_captured_node_names(future_captured_node_names, newly_planned_agents, iteration)

            return_agents_stage(
                agent, hr_agents, lr_agents, iteration, config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                newly_planned_agents, future_captured_node_names,
                self.agents, self.agents_dict, self.nodes, self.nodes_dict, self.agents_to_return_dict,
            )

        for agent in self.agents:
            if len(agent.path) - 1 == iteration - 1:
                stay_where_you_are(agent)

    def execute_next_steps(self, iteration: int, to_assert: bool = False) -> None:
        for agent in self.agents:
            agent.execute_simple_step(iteration)

    def get_preparations(self, iteration: int):
        # ---------------------------------------------- Preparations ---------------------------------------------- #
        # build blocked nodes
        config_from: Dict[str, Node] = {}
        config_to: Dict[str, Node] = {}
        goals_dict: Dict[str, Node] = {}
        curr_n_name_to_a_dict: Dict[str, AlgCgar2MapfAgent] = {}
        curr_n_name_to_a_list: List[str] = []
        for agent in self.agents:
            config_from[agent.name] = agent.curr_node
            goals_dict[agent.name] = agent.get_goal_node()
            curr_n_name_to_a_dict[agent.curr_node.xy_name] = agent
            assert agent.path[iteration-1] == agent.curr_node
            heapq.heappush(curr_n_name_to_a_list, agent.curr_node.xy_name)
        # ---------------------------------------------------------------------------------------------------------- #
        return config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list


@use_profiler(save_dir='../stats/alg_cgar2_mapf.pstat')
def main():
    # single_mapf_run(AlgCgaMapf, is_SACGR=True)
    single_mapf_run(AlgCgar2Mapf, is_SACGR=False)


if __name__ == '__main__':
    main()



















