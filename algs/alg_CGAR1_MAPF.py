from algs.alg_CGAR1_MAPF_functions import *


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
        self.agents: List[AlgCgarMapfAgent] = []
        self.agents_dict: Dict[str, AlgCgarMapfAgent] = {}
        self.agents_num_dict: Dict[int, AlgCgarMapfAgent] = {}
        self.need_to_freeze_main_goal_node: bool = True
        self.agents_to_return: List[AlgCgarMapfAgent] = []
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
        solved: List[AlgCgarMapfAgent] = [a for a in self.agents if a.goal_node == a.curr_node]
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
            new_agent = AlgCgarMapfAgent(num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes,
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
        - while not everyone are at their goals_dict:
            - PREPARATIONS
            - UPDATE PRIORITIES:
                <IDEA: we ensure that every iteration there is a CGAR instance for the first agent>
                - if you need to wait for others to complete -> return
                - update goals_dict
                - order everyone so that those with free goal positions will be first
                - check if the goal location is empty and achievable
                - if not empty:
                    - send the agent at the goal to some other location that is free and not goal of anyone
                    - set this agent to be first
                - if not achievable:
                    - reset itself to some other non-goal free location
            - CALCULATE NEST STEPS:
                <IDEA: only first agent really runs CGAR. The rest execute simple PIBT or EP steps>
                - move main agent
                - move return agents
                - move others (PIBT or EP steps)
                - stay on the spot if not planned
            - EXECUTE STEPS:
                - update curr_node of every agent to the relevant location in the path
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
            # current step iteration
            iteration += 1
            # PREPARE
            # curr_n_name_to_agent_dict, curr_n_name_to_agent_list
            config_from, config_to, goals_dict, curr_n_name_to_agent_dict, curr_n_name_to_agent_list = self.get_preparations(iteration)
            # UPDATE ORDER
            self.update_priorities(goals_dict, curr_n_name_to_agent_dict, curr_n_name_to_agent_list, iteration, to_assert)
            # CALC NEXT STEP
            self.all_calc_next_steps(config_from, config_to, goals_dict, curr_n_name_to_agent_dict, curr_n_name_to_agent_list, iteration, to_assert)
            # EXECUTE THE STEP
            self.all_execute_next_steps(iteration, to_assert)


            # PRINT
            runtime = time.time() - start_time
            print(f'\r{'*' * 20} | [{self.name}] {iteration=} | solved: {self.n_solved}/{self.n_agents} | runtime: {runtime: .2f} seconds | {'*' * 20}', end='')
            # RENDER
            if to_render and iteration >= 0:
                i_agent = self.agents[0]
                non_sv_nodes_np = self.non_sv_nodes_with_blocked_np[i_agent.get_goal_node().x, i_agent.get_goal_node().y]
                plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': i_agent, 'non_sv_nodes_np': non_sv_nodes_np}
                plot_step_in_env(ax[0], plot_info)
                plot_return_paths(ax[1], plot_info)
                plt.pause(plot_rate)
            # ASSERT
            if to_assert:
                check_vc_ec_neic_iter(self.agents, iteration)

        if to_assert:
            print(f'\n')
            max_len = align_all_paths(self.agents)
            for i in range(max_len):
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

    def get_preparations(self, iteration: int):
        # ---------------------------------------------- Preparations ---------------------------------------------- #
        # build blocked nodes
        config_from: Dict[str, Node] = {}
        config_to: Dict[str, Node] = {}
        goals_dict: Dict[str, Node] = {}
        curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent] = {}
        curr_n_name_to_agent_list: List[str] = []
        for agent in self.agents:
            config_from[agent.name] = agent.curr_node
            goals_dict[agent.name] = agent.get_goal_node()
            curr_n_name_to_agent_dict[agent.curr_node.xy_name] = agent
            assert agent.path[iteration-1] == agent.curr_node
            heapq.heappush(curr_n_name_to_agent_list, agent.curr_node.xy_name)
        # ---------------------------------------------------------------------------------------------------------- #
        return config_from, config_to, goals_dict, curr_n_name_to_agent_dict, curr_n_name_to_agent_list

    def update_priorities(
            self, goals_dict: Dict[str, Node], curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
            curr_n_name_to_agent_list: List[str], iteration: int, to_assert: bool = False
    ) -> None:
        """
        - UPDATE PRIORITIES:
                <IDEA: we ensure that every iteration there is a CGAR instance for the first agent>
                - if you need to wait for others to complete -> return
                - update goals
                - order everyone so that those with free goal positions will be first
                - check if the goal location is empty and achievable
                - if not empty:
                    - send the agent at the goal to some other location that is free and not goal of anyone
                    - set this agent to be first
                - if not achievable:
                    - reset itself to some other non-goal free location
        """
        prev_main_agent = self.agents[0]
        if len(self.agents_to_return) > 0:
            # need to wait
            goal_location_is_occupied, distur_a = get_goal_location_is_occupied(
                prev_main_agent, curr_n_name_to_agent_dict, curr_n_name_to_agent_list)
            assert not goal_location_is_occupied
            return

        if prev_main_agent.alt_goal_node is not None and prev_main_agent.curr_node == prev_main_agent.alt_goal_node:
            prev_main_agent.remove_alt_goal_node()

        init_len = len(self.agents)

        unfinished: List[AlgCgarMapfAgent] = [a for a in self.agents if a.curr_node != a.get_goal_node()]
        # random.shuffle(unfinished)
        blocked_names = [n.xy_name for n in get_blocked_nodes_for_ev(self.agents, iteration)]
        no_no_list: List[str] = [*curr_n_name_to_agent_list, *blocked_names]
        goal_free_list: List[AlgCgarMapfAgent] = [a for a in unfinished if a.get_goal_node().xy_name not in no_no_list]
        not_goal_free_list: List[AlgCgarMapfAgent] = [a for a in unfinished if a.get_goal_node().xy_name in no_no_list]
        finished: List[AlgCgarMapfAgent] = [a for a in self.agents if a.curr_node == a.get_goal_node()]
        random.shuffle(finished)

        # self.agents = [*unfinished, *finished]
        self.agents = [*goal_free_list, *not_goal_free_list, *finished]

        update_priority_numbers(self.agents)
        self.agents = reset_the_first_agent_if_goal_occupied(
            self.agents, self.nodes_dict, self.h_dict, curr_n_name_to_agent_dict, curr_n_name_to_agent_list,
            self.non_sv_nodes_with_blocked_np, iteration
        )
        update_priority_numbers(self.agents)

        curr_main_agent = self.agents[0]
        goal_location_is_occupied, distur_a = get_goal_location_is_occupied(
            curr_main_agent, curr_n_name_to_agent_dict, curr_n_name_to_agent_list)
        assert not goal_location_is_occupied
        if curr_main_agent != prev_main_agent:
            print(f'\n --- main agent: {self.agents[0].name} ---')
        assert len(set(self.agents)) == init_len
        return

    def add_to_agents_to_return(self, new_agents: List[AlgCgarMapfAgent]) -> None:
        for a in new_agents:
            if a not in self.agents_to_return:
                self.agents_to_return.append(a)

    def clean_agents_to_return(self, iteration: int) -> None:
        cleaned_agents_to_return: List[AlgCgarMapfAgent] = []
        deleted_agents: List[AlgCgarMapfAgent] = []
        for agent in self.agents_to_return:
            assert len(agent.path) - 1 >= iteration
            if len(agent.return_road) == 1 and len(agent.path[iteration:]) == 1:
                assert agent.return_road[-1][3] == agent.path[iteration]
                deleted_agents.append(agent)
            else:
                cleaned_agents_to_return.append(agent)
        self.agents_to_return = cleaned_agents_to_return
        for da in deleted_agents:
            da.reset_return_road()

    def all_calc_next_steps(
            self, config_from, config_to, goals, curr_n_name_to_agent_dict, curr_n_name_to_agent_list,
            iteration: int, to_assert: bool = False
    ) -> Tuple[List[AlgCgarMapfAgent], List[AlgCgarMapfAgent], List[AlgCgarMapfAgent]]:

        # ---------------------------------------------------------------------------------------------------------- #
        # assertions
        # ---------------------------------------------------------------------------------------------------------- #
        main_agent = self.agents[0]
        goal_location_is_occupied, distur_a = get_goal_location_is_occupied(main_agent, curr_n_name_to_agent_dict, curr_n_name_to_agent_list)
        assert not goal_location_is_occupied

        # ---------------------------------------------------------------------------------------------------------- #
        # decide for the main agent
        # ---------------------------------------------------------------------------------------------------------- #
        # all except the main agent (the first one)
        agents_without_plan: List[AlgCgarMapfAgent] = [a for a in self.agents[1:] if len(a.path) - 1 == iteration - 1]
        main_agent_decision_message = main_agent_decision(
            main_agent, self.agents, self.agents_dict, self.nodes, self.nodes_dict, self.h_dict,
            config_from, config_to, goals, curr_n_name_to_agent_dict, curr_n_name_to_agent_list,
            self.non_sv_nodes_with_blocked_np, iteration, to_assert
        )
        agents_with_newly_created_plan: List[AlgCgarMapfAgent] = [a for a in agents_without_plan if len(a.path) - 1 >= iteration]
        assert main_agent not in agents_with_newly_created_plan
        # print(f'\n before: {len(agents_without_plan)} -> after: {len(agents_with_newly_created_plan)}')
        # ---------------------------------------------------------------------------------------------------------- #
        # plan for backstep agents
        # ---------------------------------------------------------------------------------------------------------- #
        # Preps
        self.add_to_agents_to_return(agents_with_newly_created_plan)
        agents_with_prev_plan: List[AlgCgarMapfAgent] = [a for a in self.agents_to_return if a not in agents_with_newly_created_plan and len(a.path) - 1 >= iteration]
        planned_agents = [*agents_with_newly_created_plan, *agents_with_prev_plan]
        from_n_to_a_dict = curr_n_name_to_agent_dict
        backward_step_agents = [a for a in self.agents_to_return if a not in planned_agents]
        for ba in backward_step_agents:
            assert len(ba.path) - 1 == iteration - 1
        future_captured_node_names: List[str] = [n.xy_name for n in get_blocked_nodes_for_ev(self.agents, iteration)]
        fs_to_a_dict = {a.path[iteration].xy_name: a for a in self.agents if len(a.path) - 1 >= iteration}
        to_config = {a.name: a.path[iteration] for a in self.agents if len(a.path) - 1 >= iteration}

        all_update_return_roads(planned_agents, iteration)
        calc_backward_road(
            backward_step_agents, planned_agents, self.agents_to_return, self.agents_dict, from_n_to_a_dict,
            future_captured_node_names, fs_to_a_dict, to_config, iteration, to_assert
        )
        self.clean_agents_to_return(iteration)
        # for new_a in agents_with_newly_created_plan:
        #     assert new_a in self.agents_to_return
        # ---------------------------------------------------------------------------------------------------------- #
        # if at your goal location - stay
        # ---------------------------------------------------------------------------------------------------------- #
        # all_others_if_at_goal_then_stay(self.agents, main_agent, iteration, to_assert)
        # ---------------------------------------------------------------------------------------------------------- #
        # to plan for rest of the agents
        # ---------------------------------------------------------------------------------------------------------- #
        for agent in self.agents[1:]:
            # already planned
            if len(agent.path) - 1 >= iteration:
                continue
            # already at its goal
            if agent.curr_node == agent.get_goal_node():
                # if agent.curr_node == agent.goal_node:
                continue
            r_message = regular_agent_decision(
                agent, self.agents, self.agents_dict, self.nodes, self.nodes_dict, self.h_dict,
                config_from, config_to, goals, curr_n_name_to_agent_dict, curr_n_name_to_agent_list,
                self.agents_to_return, self.non_sv_nodes_with_blocked_np, self.need_to_freeze_main_goal_node,
                iteration, to_assert)

        # ---------------------------------------------------------------------------------------------------------- #
        # if no decision - just stay
        # ---------------------------------------------------------------------------------------------------------- #
        for agent in self.agents:
            # assert len(agent.path) - 1 >= iteration
            if len(agent.path) - 1 == iteration - 1:
                agent.path.append(agent.path[-1])
        # ---------------------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------------------- #
        # return forward_step_agents, backward_step_agents, other_agents
        return [], [], []

    def all_execute_next_steps(self, iteration: int, to_assert: bool = False) -> Dict[str, AlgCgarMapfAgent]:
        fs_to_a_dict: Dict[str, AlgCgarMapfAgent] = {}
        for agent in self.agents:
            agent.execute_simple_step(iteration)
            fs_to_a_dict[agent.curr_node.xy_name] = agent

        for a in self.agents:
            assert len(a.path) - 1 >= iteration
            assert a.path[iteration] == a.curr_node
        return fs_to_a_dict


@use_profiler(save_dir='../stats/alg_cgar_mapf.pstat')
def main():
    # single_mapf_run(AlgCgaMapf, is_SACGR=True)
    single_mapf_run(AlgCgarMapf, is_SACGR=False)


if __name__ == '__main__':
    main()

#                                      131
# next:   ['18_19', '17_19', '17_20', '17_19', '17_18', '18_18', '18_18', '17_18']
# distur: ['17_22', '16_22', '17_22', '17_21', '17_21', '17_20', '17_19', '17_19']


# a_next_node = get_min_h_nei_node(agent.curr_node, given_goal_node, self.nodes_dict, self.h_dict)
# if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
#     # calc single PIBT step
#     if agent.priority == 0:
#         print(f'\npibt step')
#     # blocked_nodes, planned_agents = update_blocked_nodes(self.agents, iteration, blocked_nodes, planned_agents)
#     blocked_nodes, _ = get_blocked_nodes(self.agents, iteration)
#     self.calc_pibt_step(agent, given_goal_node, blocked_nodes, config_from, config_to, goals_dict,
#                         curr_n_name_to_agent_dict, curr_n_name_to_agent_list, iteration, to_assert=to_assert)
# else:
#     # calc evacuation of agents from the corridor
#     if agent.priority == 0:
#         print(f'\nev step')
#     self.calc_ep_steps(agent, given_goal_node, config_from, a_non_sv_nodes_np, iteration,
#                        to_assert=to_assert)



# def calc_road_to_goal(next_node: Node, goal_node: Node, nodes_dict: dict, h_dict: dict) -> List[Node]:
#     full_path = [next_node]
#     while next_node != goal_node:
#         next_node = get_min_h_nei_node(next_node, goal_node, nodes_dict, h_dict)
#         full_path.append(next_node)
#     return full_path
#
#
# def agents_on_road_moving[T](
#         given_a: T, given_goal: Node, nodes_dict: dict, h_dict: dict, curr_n_name_to_agent_dict: Dict[str, T], curr_n_name_to_agent_list: List[str]
# ) -> Tuple[bool, bool]:
#     road = calc_road_to_goal(given_a.curr_node, given_goal, nodes_dict, h_dict)
#     there_are_agents_on_road = False
#     for n in road:
#         if n.xy_name in curr_n_name_to_agent_list:
#             other_a = curr_n_name_to_agent_dict[n.xy_name]
#             if other_a != given_a:
#                 continue
#             there_are_agents_on_road = True
#             if other_a.is_moving:
#                 return there_are_agents_on_road, True
#     return there_are_agents_on_road, False


# def update_blocked_nodes[T](
#         agents: T, iteration: int, blocked_nodes: List[Node], planned_agents: List[T]
# ) -> Tuple[List[Node], List[T]]:
#     for agent in agents:
#         if agent in planned_agents:
#             continue
#         # assert len(agent.path) - 1 == iteration - 1
#         if len(agent.path) > iteration:
#             planned_agents.append(agent)
#             # heapq.heappush(planned_agents, agent)
#             for n in agent.path[iteration:]:
#                 heapq.heappush(blocked_nodes, n)
#     # assert len(blocked_nodes_names) == 0
#     return blocked_nodes, planned_agents


# ---------------------------------------------------------------------------------------------------- #
# Liberate the goal location and freeze
# ---------------------------------------------------------------------------------------------------- #
# goal_location_is_occupied, distur_a = get_goal_location_is_occupied(agent, curr_n_name_to_agent_dict,
#                                                                     curr_n_name_to_agent_list)
# to wait until the end of distur agent's path?..