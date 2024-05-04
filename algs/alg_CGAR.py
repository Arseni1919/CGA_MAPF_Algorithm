from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt


def build_corridor[T](main_agent: T, nodes_dict: Dict[str, Node], h_dict: Dict[str, np.ndarray], non_sv_nodes_np: np.ndarray) -> List[Node]:
    main_goal_node = main_agent.goal_node
    i_curr_node = main_agent.curr_node
    main_next_node = get_min_h_nei_node(i_curr_node, main_goal_node, nodes_dict, h_dict)
    assert non_sv_nodes_np[main_next_node.x, main_next_node.y] == 0
    corridor: List[Node] = [i_curr_node, main_next_node]
    while non_sv_nodes_np[main_next_node.x, main_next_node.y] == 0 and main_next_node != main_goal_node:
        main_next_node = get_min_h_nei_node(main_next_node, main_goal_node, nodes_dict, h_dict)
        corridor.append(main_next_node)
    return corridor


def unfold_path(next_node: Node, son_to_father_dict: Dict[str, Node | None]) -> List[Node]:
    path = [next_node]
    father = son_to_father_dict[next_node.xy_name]
    while father is not None:
        path.append(father)
        father = son_to_father_dict[father.xy_name]
    path.reverse()
    return path


def find_ev_path[T](curr_node: Node, corridor: List[Node], nodes_dict: Dict[str, Node], blocked_nodes: List[Node],
                    captured_free_nodes: List[Node],
                    curr_n_name_to_a_dict: Dict[str, T], curr_n_name_to_a_list: List[str]
                    ) -> Tuple[List[Node], Node]:
    open_list: Deque[Node] = deque([curr_node])
    open_names_list_heap = [curr_node.xy_name]
    heapq.heapify(open_names_list_heap)
    closed_names_list_heap = [n.xy_name for n in blocked_nodes]
    heapq.heapify(closed_names_list_heap)

    son_to_father_dict: Dict[str, Node | None] = {curr_node.xy_name: None}
    iteration: int = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.popleft()
        open_names_list_heap.remove(next_node.xy_name)
        if next_node not in corridor and next_node not in captured_free_nodes and next_node.xy_name not in curr_n_name_to_a_list:
            ev_path = unfold_path(next_node, son_to_father_dict)
            return ev_path, next_node

        for nei_name in next_node.neighbours:
            # self ref
            if nei_name == next_node.xy_name:
                continue
            if nei_name in open_names_list_heap:
                continue
            if nei_name in closed_names_list_heap:
                continue
            nei_node = nodes_dict[nei_name]
            if nei_node in blocked_nodes:
                continue
            open_list.append(nei_node)
            heapq.heappush(open_names_list_heap, nei_name)
            son_to_father_dict[nei_name] = next_node
        heapq.heappush(closed_names_list_heap, next_node.xy_name)

    raise RuntimeError('no way')


def get_last_visit_dict[T](given_list: List[Node], given_agents: List[T], iteration: int) -> Dict[str, int]:
    last_visit_dict = {n.xy_name: 0 for n in given_list}
    for m_agent in given_agents:
        for i_n, n in enumerate(m_agent.path[iteration:]):
            if n in given_list:
                last_visit_dict[n.xy_name] = max(i_n + iteration, last_visit_dict[n.xy_name])
    return last_visit_dict


def push_ev_agents[T](ev_path: List[Node], curr_n_name_to_a_dict: Dict[str, T], curr_n_name_to_a_list: List[str]) -> Tuple[int, List[T]]:
    assert ev_path[0].xy_name in curr_n_name_to_a_list
    assert ev_path[-1].xy_name not in curr_n_name_to_a_list

    ev_chain_dict: Dict[str, Node | None] = {}
    max_i = len(ev_path)
    for i, n in enumerate(ev_path):
        if i + 1 == max_i:
            ev_chain_dict[n.xy_name] = None
            break
        ev_chain_dict[n.xy_name] = ev_path[i + 1]
    assert len(ev_chain_dict) == max_i

    agents_to_assign: List[T] = []
    locations_to_assign: List[Node] = []
    for n in ev_path:
        if n.xy_name in curr_n_name_to_a_list:
            i_agent: T = curr_n_name_to_a_dict[n.xy_name]
            agents_to_assign.append(i_agent)
            locations_to_assign.append(n)
    locations_to_assign = locations_to_assign[1:]
    locations_to_assign.append(ev_path[-1])
    agents_to_assign.reverse()
    locations_to_assign.reverse()

    for a, final_n in zip(agents_to_assign, locations_to_assign):
        new_path: List[Node] = []
        curr_node: Node = a.path[-1]
        while curr_node != final_n:
            next_node: Node = ev_chain_dict[curr_node.xy_name]
            assert next_node.xy_name in curr_node.neighbours
            new_path.append(next_node)
            curr_node = next_node
        assert a.path[-1].xy_name in new_path[0].neighbours
        a.path.extend(new_path)
    max_len = max([len(a.path) for a in agents_to_assign])
    return max_len, agents_to_assign


def extend_other_paths[T](max_len: int, main_agent: T, agents: List[T]) -> None:
    for agent in agents:
        if agent == main_agent:
            continue
        while len(agent.path) < max_len:
            agent.path.append(agent.path[-1])


def push_main_agent[T](main_agent: T, corridor: List[Node], moved_agents: List[T], iteration: int) -> None:
    assert main_agent not in moved_agents
    assert len(main_agent.path) == iteration
    corridor_last_visit_count_dict = get_last_visit_dict(corridor, moved_agents, iteration)
    assert corridor[0] == main_agent.path[-1]
    prev_n = corridor[0]
    for c_n in corridor[1:]:
        c_n_last_visit = corridor_last_visit_count_dict[c_n.xy_name]
        while len(main_agent.path) <= c_n_last_visit:
            main_agent.path.append(prev_n)
        main_agent.path.append(c_n)
        prev_n = c_n


def execute_backward_steps[T](backward_step_agents: List[T], future_captured_node_names: List[str], returns_dict: Dict[str, List[str]], agents_dict: Dict[str, T], iteration: int) -> None:
    for agent in backward_step_agents:
        assert len(agent.path) == iteration
    config_from: Dict[str, Node] = {a.name: a.path[-1] for a in backward_step_agents}
    config_to: Dict[str, Node] = {}

    for agent in backward_step_agents:
        from_node: Node = config_from[agent.name]
        assert len(agent.return_nodes) != 0
        if len(agent.return_nodes) == 1:
            assert agent.return_nodes[0] == agent.goal_node
            config_to[agent.name] = from_node
            continue
        assert from_node == agent.return_nodes[-1]
        next_possible_node = agent.return_nodes[-2]
        next_possible_node_name = next_possible_node.xy_name
        if next_possible_node_name in future_captured_node_names:
            config_to[agent.name] = from_node
            continue
        next_possible_n_visitors = returns_dict[next_possible_node_name]
        assert agent.name in next_possible_n_visitors
        assert len(next_possible_n_visitors) != 0
        if len(next_possible_n_visitors) == 1:
            assert next_possible_n_visitors[0] == agent.name
            agent.return_nodes = agent.return_nodes[:-1]
            config_to[agent.name] = next_possible_node
            continue
        last_two_visitors_names = next_possible_n_visitors[-2:]
        if len(last_two_visitors_names) >= 2 and agent.name not in last_two_visitors_names:
            config_to[agent.name] = from_node
            continue

    config_to_n_names: List[str] = [n.xy_name for n in config_to.values()]
    heapq.heapify(config_to_n_names)
    config_from_n_names: List[str] = [n.xy_name for n in config_from.values()]
    heapq.heapify(config_from_n_names)
    from_n_to_a_dict = {v.xy_name: k for k, v in config_from.items()}
    assert len(from_n_to_a_dict) == len(config_from)

    for agent in backward_step_agents:
        if agent.name not in config_to:
            next_possible_node = agent.return_nodes[-2]
            next_possible_node_name = next_possible_node.xy_name
            config_to[agent.name] = next_possible_node

    there_are_collisions = True
    while there_are_collisions:
        there_are_collisions = False
        for agent1, agent2 in combinations(backward_step_agents, 2):
            a1_from_node = config_from[agent1.name]
            a1_to_node = config_to[agent1.name]
            a2_from_node = config_from[agent2.name]
            a2_to_node = config_to[agent2.name]
            if a1_to_node == a2_to_node:
                config_to[agent1.name] = a1_from_node
                config_to[agent2.name] = a2_from_node
                there_are_collisions = True
                break
            if a1_from_node == a2_to_node and a1_to_node == a2_from_node:
                config_to[agent1.name] = a1_from_node
                config_to[agent2.name] = a2_from_node
                there_are_collisions = True
                break

    for agent in backward_step_agents:
        a_from_node = config_from[agent.name]
        a_to_node = config_to[agent.name]
        if a_from_node != a_to_node:
            next_n_visitors = returns_dict[a_from_node.xy_name]
            next_n_visitors.remove(agent.name)
            if len(agent.return_nodes) >= 2 and agent.return_nodes[-1] == a_from_node and agent.return_nodes[-2] == a_to_node:
                agent.return_nodes = agent.return_nodes[:-1]


    # update paths + execute the step
    for agent in backward_step_agents:
        assert agent.path[-1].xy_name in config_to[agent.name].neighbours
        next_node = config_to[agent.name]
        agent.path.append(next_node)
        agent.prev_node = agent.curr_node
        agent.curr_node = next_node

    # while len(config_to) < len(config_from):
    #     for agent in backward_step_agents:
    #         from_node = config_from[agent.name]
    #         assert agent.return_nodes[-1] == from_node
    #         next_possible_node = agent.return_nodes[-2]
    #         next_possible_node_name = next_possible_node.xy_name
    #         next_possible_n_visitors = returns_dict[next_possible_node_name]
    #         if next_possible_node_name in config_to_n_names:
    #             config_to[agent.name] = from_node
    #             heapq.heappush(config_to_n_names, from_node.xy_name)
    #             break
    #
    #
    #
    #
    # for agent in backward_step_agents:
    #     if agent.name not in config_to:
    #         # config_to[agent.name] = agent.path[-1]
    #         open_list: Deque[T] = deque([agent])
    #         while len(open_list) > 0:
    #             next_agent = open_list.popleft()
    #             next_possible_node = next_agent.return_nodes[-2]
    #             next_possible_node_name = next_possible_node.xy_name
    #             next_possible_n_visitors = returns_dict[next_possible_node_name]
    #             assert next_agent.name in next_possible_n_visitors[-2:]
    #             if next_possible_node_name in config_to_n_names:
    #                 next_node: Node = next_agent.path[-1]
    #                 config_to[next_agent.name] = next_node
    #                 heapq.heappush(config_to_n_names, next_node.xy_name)
    #                 continue
    #             if next_possible_node_name not in config_from_n_names:
    #                 next_agent.return_nodes = next_agent.return_nodes[:-1]
    #                 config_to[next_agent.name] = next_possible_node
    #                 heapq.heappush(config_to_n_names, next_possible_node_name)
    #                 next_possible_n_visitors.remove(next_agent.name)
    #                 continue
    #
    #             agent_to_push_name = from_n_to_a_dict[next_possible_node_name]
    #             agent_to_push = agents_dict[agent_to_push_name]
    #             if agent_to_push_name in config_to:
    #                 assert config_to[agent_to_push_name] != next_possible_node
    #                 next_agent.return_nodes = next_agent.return_nodes[:-1]
    #                 config_to[next_agent.name] = next_possible_node
    #                 heapq.heappush(config_to_n_names, next_possible_node_name)
    #                 next_possible_n_visitors.remove(next_agent.name)
    #                 continue
    #             open_list.appendleft(next_agent)
    #             open_list.appendleft(agent_to_push)


            # next_possible_n_visitors.pop()
            # self.return_nodes = self.return_nodes[:-1]
            # self.path.append(next_possible_node)


class AlgCGARAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.path: List[Node] = [start_node]
        self.first_arrived: bool = self.curr_node == self.goal_node
        self.return_nodes: List[Node] = [self.goal_node]
        self.arrived: bool = self.first_arrived

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def return_nodes_names(self):
        return [n.xy_name for n in self.return_nodes]

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
        return self.goal_node.xy_name

    @property
    def a_start_node_name(self):
        return self.start_node.xy_name

    def __eq__(self, other):
        return self.num == other.num

    def __hash__(self):
        return hash(self.name)

    def execute_forward_step(self, iteration: int, returns_dict: Dict[str, List[str]]) -> None:
        # execute the step
        next_node = self.path[iteration]
        self.prev_node = self.curr_node
        self.curr_node = next_node

        # for the back-steps
        if self.curr_node == self.goal_node:
            self.first_arrived = True
        if self.first_arrived:
            if self.curr_node != self.return_nodes[-1]:
                assert self.curr_node.xy_name in self.return_nodes[-1].neighbours
                self.return_nodes.append(self.curr_node)
                returns_dict[self.curr_node.xy_name].append(self.name)
                # returns_dict[self.prev_node.xy_name].append(self.name)

    def execute_backward_step(self, iteration: int, returns_dict: Dict[str, List[str]], config_to_node_names: List[str]) -> None:
        assert len(self.return_nodes) != 0
        if len(self.return_nodes) == 1:
            assert self.return_nodes[0] == self.goal_node
            self.path.append(self.path[-1])
        else:
            assert self.curr_node == self.return_nodes[-1]
            next_possible_node = self.return_nodes[-2]
            next_possible_node_name = next_possible_node.xy_name
            if next_possible_node.xy_name in config_to_node_names:
                self.path.append(self.path[-1])
            else:
                next_possible_n_visitors = returns_dict[next_possible_node.xy_name]
                if len(next_possible_n_visitors) == 0:
                    assert next_possible_node == self.goal_node
                    self.return_nodes = self.return_nodes[:-1]
                    self.path.append(next_possible_node)
                    assert self.return_nodes[-1] == self.path[-1]
                else:
                    last_visitor_name = next_possible_n_visitors[-1]
                    if last_visitor_name == self.name:
                        next_possible_n_visitors.pop()
                        self.return_nodes = self.return_nodes[:-1]
                        self.path.append(next_possible_node)
                    else:
                        self.path.append(self.path[-1])

        # execute the step
        next_node = self.path[iteration]
        self.prev_node = self.curr_node
        self.curr_node = next_node


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
        self.returns_dict: Dict[str, List[str]] = {n.xy_name: [] for n in self.nodes}

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
        self.returns_dict: Dict[str, List[str]] = {n.xy_name: [] for n in self.nodes}
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgCGARAgent(num=num, start_node=start_node, goal_node=goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            if new_agent.start_node == new_agent.goal_node:
                self.returns_dict[new_agent.goal_node.xy_name].append(new_agent.name)

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
            # plot_rate = 2

        blocked_nodes = [self.main_agent.goal_node]
        self.non_sv_nodes_np = get_non_sv_nodes_np(self.nodes, self.nodes_dict, self.img_np, blocked_nodes=blocked_nodes)

        iteration = 0
        while not self.stop_condition():
            if len(self.main_agent.path) - 1 < iteration:

                # assert
                if to_assert:
                    for agent in self.agents:
                        assert len(agent.path) == iteration

                # if you are here, there is a need for a plan for a future step
                main_next_node = get_min_h_nei_node(self.main_agent.curr_node, self.main_agent.goal_node, self.nodes_dict, self.h_dict)
                if self.non_sv_nodes_np[main_next_node.x, main_next_node.y]:
                    # calc single PIBT step
                    self.calc_pibt_step(iteration, blocked_nodes)
                else:
                    # calc evacuation of agents from the corridor
                    self.calc_ep_steps(iteration, blocked_nodes)

            future_captured_node_names: List[str] = []
            forward_step_agents: List[AlgCGARAgent] = []
            backward_step_agents: List[AlgCGARAgent] = []
            for agent in self.agents:
                if len(agent.path) - 1 >= iteration:
                    forward_step_agents.append(agent)
                    for n in agent.path[iteration:]:
                        heapq.heappush(future_captured_node_names, n.xy_name)
                else:
                    backward_step_agents.append(agent)

            # execute the step
            for agent in forward_step_agents:
                agent.execute_forward_step(iteration, self.returns_dict)

            execute_backward_steps(backward_step_agents, future_captured_node_names, self.returns_dict, self.agents_dict, iteration)

            # for agent in backward_step_agents:
            #     agent.execute_backward_step(iteration, self.returns_dict, future_captured_node_names)


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

        # Baseline reverse part
        # for agent in self.agents:
        #     if agent != self.main_agent:
        #         reverse_path = agent.path[:-1]
        #         reverse_path.reverse()
        #         agent.path.extend(reverse_path)
        #     else:
        #         agent.path.extend([agent.path[-1] for _ in range(len(agent.path) - 1)])
        #
        # while iteration < len(self.main_agent.path):
        #     # execute the step
        #     for agent in self.agents:
        #         next_node = agent.path[iteration]
        #         agent.prev_node = agent.curr_node
        #         agent.curr_node = next_node
        #
        #     # updates after the step execution
        #     iteration += 1
        #
        #     # print + render
        #     print(f'\r[CGAR] {iteration=} | ', end='')
        #
        #     if to_render and iteration >= 0:
        #         # i_agent = self.agents_dict['agent_0']
        #         i_agent = self.agents[0]
        #         plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': i_agent, }
        #         plot_step_in_env(ax[0], plot_info)
        #         # plot_total_finished_goals(ax[1], plot_info)
        #         # plot_unique_movements(ax[1], plot_info)
        #         plt.pause(plot_rate)

        paths_dict = {a.name: a.path for a in self.agents}
        for agent in self.agents:
            if agent == self.main_agent:
                assert agent.path[-1] == agent.goal_node
            else:
                assert agent.path[-1] == agent.start_node
        return True, paths_dict

    def calc_pibt_step(self, iteration: int, blocked_nodes: List[Node]):
        # print(f'\n --- inside calc_pibt_step {iteration} --- ')
        assert len(self.main_agent.path) - 1 < iteration
        # Preps
        config_to = {}
        for agent in self.agents:
            if len(agent.path) - 1 >= iteration:
                config_to[agent.name] = agent.path[iteration]

        # Calc PIBT
        config_to = run_i_pibt(self.main_agent, self.agents, self.nodes_dict, self.h_dict, config_to=config_to, blocked_nodes=blocked_nodes)

        # Extend the paths
        for agent in self.agents:
            if agent.name in config_to:
                next_node = config_to[agent.name]
                agent.path.append(next_node)

    def calc_ep_steps(self, iteration: int, blocked_nodes: List[Node]):
        """
        - Build corridor
        - Build EP for ev-agents in the corridor
        - Evacuate ev-agents
        - Build the steps in the corridor to the main agent
        """
        # Build corridor
        # print(f'\n --- inside calc_ep_steps {iteration} --- ')
        corridor = build_corridor(self.main_agent, self.nodes_dict, self.h_dict, self.non_sv_nodes_np)
        # Build EvPaths (evacuation paths) for ev-agents in the corridor
        curr_n_name_to_a_dict: Dict[str, AlgCGARAgent] = {a.path[-1].xy_name: a for a in self.agents}
        curr_n_name_to_a_list: List[str] = list(curr_n_name_to_a_dict.keys())
        heapq.heapify(curr_n_name_to_a_list)
        ev_agents: List[AlgCGARAgent] = []
        assert corridor[0] == self.main_agent.path[-1]
        for node in corridor[1:]:
            if node.xy_name in curr_n_name_to_a_list:
                ev_agents.append(curr_n_name_to_a_dict[node.xy_name])

        ev_paths_list: List[List[Node]] = []
        blocked_nodes: List[Node] = [self.main_agent.path[-1], self.main_agent.goal_node]
        captured_free_nodes: List[Node] = []
        for ev_agent in ev_agents:
            ev_path, captured_free_node = find_ev_path(
                ev_agent.curr_node, corridor, self.nodes_dict, blocked_nodes, captured_free_nodes,
                curr_n_name_to_a_dict, curr_n_name_to_a_list
            )
            captured_free_nodes.append(captured_free_node)
            ev_paths_list.append(ev_path)

        # Build steps for the ev-agents inside the EvPaths + extend paths
        moved_agents = []
        for i_ev_path, ev_path in enumerate(ev_paths_list):
            curr_n_name_to_a_dict: Dict[str, AlgCGARAgent] = {a.path[-1].xy_name: a for a in self.agents}
            curr_n_name_to_a_list: List[str] = list(curr_n_name_to_a_dict.keys())
            max_len, assigned_agents = push_ev_agents(ev_path, curr_n_name_to_a_dict, curr_n_name_to_a_list)
            assert self.main_agent not in assigned_agents
            # extend_other_paths(max_len, self.main_agent, self.agents)
            moved_agents.extend(assigned_agents)
            moved_agents = list(set(moved_agents))

        # Build the steps in the corridor to the main agent + extend the path
        push_main_agent(self.main_agent, corridor, moved_agents, iteration)
        # total_max_len = max([len(a.path) for a in self.agents])

    def stop_condition(self):
        for agent in self.agents:
            if agent.path[-1] != agent.goal_node:
                return False
        return True

@use_profiler(save_dir='../stats/alg_cgar.pstat')
def main():
    single_mapf_run(AlgCGAR, is_SACGR=True)


if __name__ == '__main__':
    main()
