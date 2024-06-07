import random

from algs.alg_CGAR3_MAPF_functions import *


class AlgCgar3Mapf(AlgGeneric):

    name = 'CGAR3-MAPF'

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
        self.agents: List[AlgCgar3MapfAgent] = []
        self.agents_dict: Dict[str, AlgCgar3MapfAgent] = {}
        self.agents_num_dict: Dict[int, AlgCgar3MapfAgent] = {}
        self.r_parent_to_children_dict: Dict[str, List[AlgCgar3MapfAgent]] = {}
        self.r_children_to_parent_dict: Dict[str, AlgCgar3MapfAgent | None] = {}
        self.n_agents = 0
        self.with_return_stage = True
        # self.with_return_stage = False
        # self.first_privilege = True
        self.first_privilege = False

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
        solved: List[AlgCgar3MapfAgent] = [a for a in self.agents if a.goal_node == a.curr_node]
        return len(solved)

    @property
    def r_parent_to_children_dict_names(self):
        r_parent_to_children_dict_names = {}
        for k, v in self.r_parent_to_children_dict.items():
            new_v = [a.name for a in v]
            r_parent_to_children_dict_names[k] = new_v
        return r_parent_to_children_dict_names

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
            new_agent = AlgCgar3MapfAgent(
                num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes, nodes_dict=self.nodes_dict
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent
            self.r_parent_to_children_dict[new_agent.name] = []
            self.r_children_to_parent_dict[new_agent.name] = None
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
            # plot_rate = 1
            # plot_rate = 2

        iteration = 0  # all at their start locations
        while not self.stop_condition():
            # next step iteration
            iteration += 1
            # UPDATE ORDER
            self.update_priorities(iteration, to_assert)
            # BUILD NEXT STEP
            config_to: Dict[str, Node] = self.build_next_steps(iteration, to_assert)
            # EXECUTE THE STEP
            self.execute_next_steps(config_to, iteration, to_assert)

            # PRINT
            runtime = time.time() - start_time
            print(f'\r{'*' * 20} | [{self.name}] {iteration=} | solved: {self.n_solved}/{self.n_agents} | runtime: {runtime: .2f} seconds | {'*' * 20}', end='')
            # RENDER
            if to_render and iteration >= 45:
                i_agent = self.agents[0]
                non_sv_nodes_np = self.non_sv_nodes_with_blocked_np[i_agent.get_goal_node().x, i_agent.get_goal_node().y]
                plot_info = {
                    'img_np': self.img_np,
                    'agents': self.agents,
                    'i_agent': i_agent,
                    'r_parent_to_children_dict': self.r_parent_to_children_dict,
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
        unfinished: List[AlgCgar3MapfAgent] = []
        finishing: List[AlgCgar3MapfAgent] = []
        finished: List[AlgCgar3MapfAgent] = []
        for agent in self.agents:
            return_agents = self.r_parent_to_children_dict[agent.name]
            if agent.curr_node == agent.get_goal_node() and len(return_agents) == 0:
                finished.append(agent)
            else:
                unfinished.append(agent)
        random.shuffle(finished)

        self.agents = [*unfinished, *finished]

        update_ranks(self.agents)
        update_status(self.agents)

    def build_next_steps(self, iteration: int, to_assert: bool = False) -> Dict[str, Node]:

        (config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list,
         blocked_map, r_blocked_map, f_blocked_map, curr_map,
         agents_with_plan, agents_with_no_plan) = self.get_preparations(iteration)
        remained_agents: List[AlgCgar3MapfAgent] = []

        for curr_rank, agent in enumerate(self.agents):
            assert agent.curr_rank == curr_rank
            pa_list: List[AlgCgar3MapfAgent] = []

            # CHECK_STAGE
            changed, check_stage_info, fresh_agents = continuation_check_stage(
                agent, blocked_map, iteration,
                config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                self.r_parent_to_children_dict, self.h_dict,
                self.non_sv_nodes_with_blocked_np, self.nodes_dict,
                self.first_privilege
            )
            pa_list.extend(fresh_agents)

            # if curr_rank == 0 and iteration >= 800:
            #     print('', end='')

            # STEP_STAGE
            calc_step_stage_message, fresh_agents = calc_step_stage(
                agent, blocked_map, r_blocked_map, f_blocked_map, curr_map, agents_with_no_plan, iteration,
                config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list, check_stage_info,
                self.non_sv_nodes_with_blocked_np, self.agents, self.nodes, self.nodes_dict, self.h_dict
            )
            pa_list.extend(fresh_agents)

            # RETURN_STAGE
            only_to_first = agent.curr_rank == 0 and self.first_privilege
            # if self.with_return_stage and iteration >= 200 and agent.curr_rank == 0:
            # if self.with_return_stage and iteration >= 200:
            # if self.with_return_stage and agent.curr_rank == 0:
            if self.with_return_stage:
                if only_to_first or not self.first_privilege:
                    # Get newly-moved agents
                    newly_planned_agents = get_newly_planned_agents(agent, pa_list, iteration)
                    f_blocked_map = update_f_blocked_map(f_blocked_map, pa_list, iteration)
                    return_stage_message, fresh_agents = return_agents_stage(
                        agent, iteration, check_stage_info,
                        config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                        newly_planned_agents, f_blocked_map,
                        self.agents, self.agents_dict, self.r_parent_to_children_dict, self.r_children_to_parent_dict
                    )
                    pa_list.extend(fresh_agents)
                # check_configs_with_agent(agent, self.agents, config_from, config_to)


            # update blocked map
            blocked_map, r_blocked_map, f_blocked_map = update_blocked_maps(
                blocked_map, r_blocked_map, f_blocked_map, agent, pa_list, self.r_parent_to_children_dict,
                self.first_privilege,
                iteration
            )
            if len(agent.path) - 1 == iteration - 1:
                remained_agents.append(agent)

        for agent in remained_agents:
            if len(agent.path) - 1 == iteration - 1:
                stay_where_you_are(agent, config_to, iteration)
        # check_configs(self.agents, config_from, config_to, final_check=True)
        return config_to

    def execute_next_steps(self, config_to: Dict[str, Node], iteration: int, to_assert: bool = False) -> None:
        for agent in self.agents:
            agent.execute_simple_step(iteration)
            assert config_to[agent.name] == agent.curr_node

    def get_preparations(self, iteration: int):
        # ---------------------------------------------- Preparations ---------------------------------------------- #
        # build blocked nodes
        config_from: Dict[str, Node] = {}
        config_to: Dict[str, Node] = {}
        goals_dict: Dict[str, Node] = {}
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent] = {}
        curr_n_name_to_a_list: List[str] = []
        blocked_map: np.ndarray = np.zeros(self.img_np.shape)
        r_blocked_map: np.ndarray = np.zeros(self.img_np.shape)
        f_blocked_map: np.ndarray = np.zeros(self.img_np.shape)
        curr_map: np.ndarray = np.zeros(self.img_np.shape)
        agents_with_plan: List[AlgCgar3MapfAgent] = []
        agents_with_no_plan: List[AlgCgar3MapfAgent] = []
        for agent in self.agents:
            config_from[agent.name] = agent.curr_node
            curr_map[agent.curr_node.x, agent.curr_node.y] = 1
            goals_dict[agent.name] = agent.get_goal_node()
            curr_n_name_to_a_dict[agent.curr_node.xy_name] = agent
            if len(agent.path) - 1 >= iteration:
                agents_with_plan.append(agent)
                config_to[agent.name] = agent.path[iteration]
                future_path = agent.path[iteration - 1:]
                for n in future_path:
                    blocked_map[n.x, n.y] = 1
                for n in agent.path[iteration:]:
                    f_blocked_map[n.x, n.y] = 1
            else:
                agents_with_no_plan.append(agent)
                # n = agent.path[iteration - 1]
                # heapq.heappush(future_captured_node_names, n.xy_name)
            heapq.heappush(curr_n_name_to_a_list, agent.curr_node.xy_name)

        # ---------------------------------------------------------------------------------------------------------- #
        return (config_from, config_to, goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                blocked_map, r_blocked_map, f_blocked_map, curr_map, agents_with_plan, agents_with_no_plan)


@use_profiler(save_dir='../stats/alg_cgar3_mapf.pstat')
def main():
    # single_mapf_run(AlgCgar3Mapf, is_SACGR=True)
    single_mapf_run(AlgCgar3Mapf, is_SACGR=False)


if __name__ == '__main__':
    main()

# last_n_name_to_a_list = [a.path[-1].xy_name for a in self.agents]
# goal_free_list: List[AlgCgar3MapfAgent] = [
#     a for a in unfinished if a.get_goal_node().xy_name not in last_n_name_to_a_list
# ]
# not_goal_free_list: List[AlgCgar3MapfAgent] = [
#     a for a in unfinished if a.get_goal_node().xy_name in last_n_name_to_a_list
# ]
# self.agents = [*finishing, *goal_free_list, *not_goal_free_list, *finished]


# future_captured_node_names: List[str] = []
# future_captured_node_names = update_future_captured_node_names(future_captured_node_names, self.agents, iteration)
# blocked_map_2, r_blocked_map, future_captured_node_names = init_blocked_map(self.agents, self.img_np, iteration)

# hr_agents = self.agents[:curr_rank]
# lr_agents = self.agents[curr_rank + 1:]

# blocked_map: np.ndarray = get_blocked_map(
#     agent, hr_agents, lr_agents, self.agents, self.r_parent_to_children_dict, self.img_np, iteration
# )
# assert (blocked_map == blocked_map_2).all()
# unplanned_agents: List[AlgCgar3MapfAgent] = [
#     a for a in self.agents if a.name not in config_to and a != agent
# ]

# ua_list = unplanned_agents[:]
# if agent.name not in config_to:
#     ua_list.append(agent)
# ua_list: List[AlgCgar3MapfAgent] = [a for a in self.agents if len(a.path) - 1 == iteration - 1]

# future_captured_node_names = update_future_captured_node_names(
#       future_captured_node_names, newly_planned_agents, iteration
# )

