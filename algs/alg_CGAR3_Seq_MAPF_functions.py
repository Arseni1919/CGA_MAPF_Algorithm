from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from algs.alg_PIBT import run_i_pibt
from algs.alg_CGAR import get_min_h_nei_node, find_ev_path, push_ev_agents, push_main_agent, build_corridor_from_nodes


class AlgCgar3SeqMapfAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node, nodes: List[Node], nodes_dict: Dict[str, Node]):
        self.num = num
        self.curr_rank = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.alt_goal_node: Node | None = None
        self.changer_agent_name: str | None = None
        self.prev_goal_node_names_list: List[str] = []
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.path: List[Node] = [start_node]
        self.return_road: Deque[Tuple[str, int, List[str], Node]] = deque()
        self.trash_return_road: Deque[Tuple[str, int, List[str], Node]] = deque()
        # Dictionary: node.xy_name -> [ ( (0) agent name, (1) iteration ), ...]
        self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def return_road_names(self):
        return [(tpl[0], tpl[1], tpl[2]) for tpl in self.return_road]

    @property
    def return_road_nodes(self):
        return [tpl[3] for tpl in self.return_road]

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

    def execute_simple_step(self, iteration: int) -> None:
        self.prev_node = self.curr_node
        self.curr_node = self.path[iteration]
        # assert self.prev_node.xy_name in self.curr_node.neighbours

    def get_goal_node(self) -> Node:
        if self.alt_goal_node is not None:
            return self.alt_goal_node
        return self.goal_node

    def reset_alt_goal_node(self, node: Node, setting_agent: Self) -> None:
        self.alt_goal_node = node
        self.changer_agent_name = setting_agent.name

    def remove_alt_goal_node(self) -> None:
        assert self.alt_goal_node is not None
        self.prev_goal_node_names_list.append(self.alt_goal_node.xy_name)
        self.alt_goal_node = None
        self.changer_agent_name = None

    def reset_return_road(self):
        self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}
        self.return_road: Deque[Tuple[str, int, List[str], Node]] = deque()

    def get_wl(self, node: Node, to_assert: bool = False):
        return self.waiting_table[node.xy_name]

    def get_wl_names(self, node: Node, to_assert: bool = False):
        return [tpl[0] for tpl in self.waiting_table[node.xy_name]]

    def add_to_wl(self, node: Node, agent_on_road: Self, iteration: int, to_assert: bool = False):
        if (agent_on_road.name, iteration) not in self.waiting_table[node.xy_name]:
            self.waiting_table[node.xy_name].append((agent_on_road.name, iteration))

    def remove_from_wl(self, node: Node, agent_on_road: Self, iteration: int, to_assert: bool = False):
        remove_item = (agent_on_road.name, iteration)
        self.waiting_table[node.xy_name] = [i for i in self.waiting_table[node.xy_name] if i != remove_item]


def check_configs(
        agents: List[AlgCgar3SeqMapfAgent] | Deque[AlgCgar3SeqMapfAgent],
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        final_check: bool = False
) -> None:
    if final_check:
        for agent in agents:
            assert agent.name in config_from
            assert agent.name in config_to
    for a1, a2 in combinations(agents, 2):
        if a1.name not in config_to or a2.name not in config_to:
            continue
        # vertex conf
        from_node_1: Node = config_from[a1.name]
        to_node_1: Node = config_to[a1.name]
        from_node_2: Node = config_from[a2.name]
        to_node_2: Node = config_to[a2.name]
        assert from_node_1 != from_node_2, f' vc: {a1.name}-{a2.name} in {from_node_1.xy_name}'
        assert to_node_1 != to_node_2, f' vc: {a1.name}-{a2.name} in {to_node_2.xy_name}'
        # edge conf
        edge1 = (from_node_1.x, from_node_1.y, to_node_1.x, to_node_1.y)
        edge2 = (from_node_2.x, from_node_2.y, to_node_2.x, to_node_2.y)
        assert edge1 != edge2, f'ec: {a1.name}-{a2.name} in {edge1}'
        # nei conf
        assert from_node_1.xy_name in to_node_1.neighbours, f'neic {a1.name}: {from_node_1.xy_name} not nei of {to_node_1.xy_name}'
        assert from_node_2.xy_name in to_node_2.neighbours, f'neic {a2.name}: {from_node_2.xy_name} not nei of {to_node_2.xy_name}'


def update_priority_numbers(agents: Deque[AlgCgar3SeqMapfAgent]):
    for i_priority, agent in enumerate(agents):
        agent.curr_rank = i_priority


def order_the_agents(
        agents: Deque[AlgCgar3SeqMapfAgent]
) -> Tuple[Deque[AlgCgar3SeqMapfAgent], Dict[str, AlgCgar3SeqMapfAgent], List[str]]:

    # Preps
    last_n_name_to_a_dict: Dict[str, AlgCgar3SeqMapfAgent] = {}
    last_n_name_to_a_list: List[str] = []
    for a in agents:
        last_n_name = a.path[-1].xy_name
        assert a.curr_node.xy_name == last_n_name
        last_n_name_to_a_dict[a.path[-1].xy_name] = a
        heapq.heappush(last_n_name_to_a_list, last_n_name)

    # Main
    unfinished: List[AlgCgar3SeqMapfAgent] = [a for a in agents if a.curr_node != a.get_goal_node()]
    goal_free_list: List[AlgCgar3SeqMapfAgent] = [a for a in unfinished if a.get_goal_node().xy_name not in last_n_name_to_a_list]
    not_goal_free_list: List[AlgCgar3SeqMapfAgent] = [a for a in unfinished if a.get_goal_node().xy_name in last_n_name_to_a_list]
    finished: List[AlgCgar3SeqMapfAgent] = [a for a in agents if a.curr_node == a.get_goal_node()]
    random.shuffle(finished)

    final_list: Deque[AlgCgar3SeqMapfAgent] = deque([*goal_free_list, *not_goal_free_list, *finished])
    update_priority_numbers(final_list)
    return final_list, last_n_name_to_a_dict, last_n_name_to_a_list


def is_enough_free_locations(
        curr_node: Node,
        goal_node: Node,
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        curr_n_name_to_agent_list: List[str],
        non_sv_nodes_np: np.ndarray,
        blocked_nodes: List[Node] | None = None,
        full_corridor_check: bool = False
) -> Tuple[bool, str, int, dict]:
    next_node = get_min_h_nei_node(curr_node, goal_node, nodes_dict, h_dict)
    full_path: List[Node] = [next_node]
    open_list: List[Node] = [next_node]
    open_list_names: List[str] = [next_node.xy_name]
    closed_list: List[Node] = [curr_node, goal_node]
    closed_list_names: List[str] = [n.xy_name for n in closed_list]
    heapq.heapify(closed_list_names)
    if blocked_nodes is None:
        blocked_nodes = []
    blocked_nodes_names = [n.xy_name for n in blocked_nodes]
    heapq.heapify(blocked_nodes_names)

    if next_node == goal_node:
        # if next_node in other_curr_nodes or next_node in blocked_nodes:
        if next_node.xy_name in curr_n_name_to_agent_list or next_node in blocked_nodes:
            return False, f'PROBLEM-1 - next node {next_node.xy_name} is a goal node and is occupied or blocked', 1, {
                'goal_node': goal_node.xy_name,
                'blocked_nodes': [n.xy_name for n in blocked_nodes],
                # 'other_curr_nodes': [n.xy_name for n in other_curr_nodes],
                'curr_nodes': curr_n_name_to_agent_list,
            }
        return True, f'OK-1 - next_node {next_node.xy_name} is a goal node and it is free', 0, {}

    closest_corridor: List[Node] = build_corridor_from_nodes(curr_node, goal_node, nodes_dict, h_dict, non_sv_nodes_np)
    if closest_corridor[-1] == goal_node:
        if closest_corridor[-1].xy_name in curr_n_name_to_agent_list or closest_corridor[-1] in blocked_nodes:
            return False, f'PROBLEM-2 - last corridor node {goal_node.xy_name} is a goal node and is occupied or blocked', 2, {
                'goal_node': goal_node.xy_name,
                'closest_corridor': [n.xy_name for n in closest_corridor],
                'blocked_nodes': [n.xy_name for n in blocked_nodes],
                # 'other_curr_nodes': [n.xy_name for n in other_curr_nodes],
                'curr_nodes': curr_n_name_to_agent_list,
            }

    if full_corridor_check:
        corridor_blocked_list = list(set(closest_corridor).intersection(blocked_nodes))
        if len(corridor_blocked_list) > 0:
            return False, f'PROBLEM-3 - part of the corridor is blocked: {[n.xy_name for n in corridor_blocked_list]}', 3, {
                'closest_corridor': [n.xy_name for n in closest_corridor],
                'corridor_blocked_list': [n.xy_name for n in corridor_blocked_list]
            }

    # calc maximum required free nodes
    max_required_free_nodes = 0
    assert closest_corridor[0] == curr_node
    for n in closest_corridor[1:]:
        if n.xy_name in curr_n_name_to_agent_list:
            max_required_free_nodes += 1
    if max_required_free_nodes == 0:
        return True, f'OK-2 - {max_required_free_nodes=}', 0, {
            'closest_corridor': [n.xy_name for n in closest_corridor]
        }

    # count available free locations
    free_count = 0
    touched_blocked_nodes = False
    touched_blocked_nodes_list: List[Node] = []
    while len(open_list) > 0:
        next_node = open_list.pop()
        open_list_names.remove(next_node.xy_name)
        next_node_out_of_full_path = next_node not in full_path
        next_node_is_not_occupied = next_node.xy_name not in curr_n_name_to_agent_list
        if next_node_out_of_full_path and next_node_is_not_occupied:
            free_count += 1
            if free_count >= max_required_free_nodes:
                return True, f'OK-3 - {free_count} free locations for {max_required_free_nodes=}', 0, {
                    'closest_corridor': [n.xy_name for n in closest_corridor],
                    'max_required_free_nodes': max_required_free_nodes,
                    'free_count': free_count,
                    'blocked_nodes': blocked_nodes,
                    'open_list_names': open_list_names,
                    'closed_list_names': closed_list_names,
                    'blocked_nodes_names': blocked_nodes_names,
                    'touched_blocked_nodes': touched_blocked_nodes,
                    'touched_blocked_nodes_list': touched_blocked_nodes_list
                }
        for nei_name in next_node.neighbours:
            if nei_name == next_node.xy_name:
                continue
            if nei_name in closed_list_names:
                continue
            if nei_name in open_list_names:
                continue
            if nei_name in blocked_nodes_names:
                touched_blocked_nodes = True
                touched_blocked_nodes_list.append(nei_name)
                continue
            nei_node = nodes_dict[nei_name]
            open_list.append(nei_node)
            heapq.heappush(open_list_names, nei_name)
        heapq.heappush(closed_list_names, next_node.xy_name)

    error_num = 4 if touched_blocked_nodes else 5
    return False, f'PROBLEM-{error_num} - not_enough_free_nodes', error_num, {
        'closest_corridor': [n.xy_name for n in closest_corridor],
        'max_required_free_nodes': max_required_free_nodes,
        'free_count': free_count,
        'blocked_nodes': blocked_nodes,
        'open_list_names': open_list_names,
        'closed_list_names': closed_list_names,
        'blocked_nodes_names': blocked_nodes_names,
        'touched_blocked_nodes': touched_blocked_nodes,
        'touched_blocked_nodes_list': touched_blocked_nodes_list
    }


def get_alter_goal_node(
        agent: AlgCgar3SeqMapfAgent,
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        non_sv_nodes_with_blocked_np: np.ndarray,
        curr_n_name_to_agent_list: List[str],
        blocked_nodes: List[Node],
        avoid_curr_nodes: bool = False,
        goals: List[Node] | None = None, avoid_goals: bool = False
) -> Node | None:
    open_list = deque([agent.curr_node])
    closed_list_names = []
    main_goal_node: Node = agent.goal_node
    main_goal_non_sv_np = non_sv_nodes_with_blocked_np[main_goal_node.x, main_goal_node.y]
    while len(open_list) > 0:
        alt_node: Node = open_list.popleft()

        # check the option
        not_curr_node: bool = alt_node != agent.curr_node
        non_sv_in_main: bool = main_goal_non_sv_np[alt_node.x, alt_node.y] == 1

        not_in_curr_nodes = True
        if avoid_curr_nodes:
            # not_in_curr_nodes = alt_node not in curr_nodes
            not_in_curr_nodes = alt_node.xy_name not in curr_n_name_to_agent_list

        not_in_goal_nodes = True
        if avoid_goals:
            not_in_goal_nodes = alt_node not in goals

        alt_non_sv_np = non_sv_nodes_with_blocked_np[alt_node.x, alt_node.y]
        alt_is_good, alt_message, i_error, info = is_enough_free_locations(
            agent.curr_node, alt_node, nodes_dict, h_dict, curr_n_name_to_agent_list, alt_non_sv_np,
            blocked_nodes, full_corridor_check=True
        )

        if not_curr_node and non_sv_in_main and not_in_curr_nodes and not_in_goal_nodes and alt_is_good:
            return alt_node

        for nn in alt_node.neighbours:
            if nn == alt_node.xy_name:
                continue
            if nn in closed_list_names:
                continue
            n = nodes_dict[nn]
            if n in open_list:
                continue
            if n in blocked_nodes:
                continue
            open_list.append(n)
        heapq.heappush(closed_list_names, alt_node.xy_name)
    return agent.goal_node


def cgar_conditions_check(
        main_agent: AlgCgar3SeqMapfAgent,
        last_n_name_to_a_dict: Dict[str, AlgCgar3SeqMapfAgent],
        last_n_name_to_a_list: List[str],
        non_sv_nodes_with_blocked_np: np.ndarray,
        h_dict: Dict[str, np.ndarray],
) -> bool:
    main_agent_goal = main_agent.get_goal_node()
    if main_agent_goal.xy_name in last_n_name_to_a_list:
        distur_agent = last_n_name_to_a_dict[main_agent_goal.xy_name]
        alter_goal_node = get_alter_goal_node(
            distur_agent, distur_agent.nodes_dict, h_dict, non_sv_nodes_with_blocked_np, last_n_name_to_a_list,
            [], avoid_curr_nodes=True)
        distur_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        return False

    main_non_sv_np = non_sv_nodes_with_blocked_np[main_agent_goal.x, main_agent_goal.y]
    goal_is_good, message, i_error, info = is_enough_free_locations(
        main_agent.curr_node, main_agent_goal, main_agent.nodes_dict, h_dict, last_n_name_to_a_list, main_non_sv_np,
        [], full_corridor_check=True
    )
    if not goal_is_good:
        alter_goal_node = get_alter_goal_node(
            main_agent, main_agent.nodes_dict, h_dict, non_sv_nodes_with_blocked_np, last_n_name_to_a_list,
            [], avoid_curr_nodes=True)
        main_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        return True
    return True


def stay_where_you_are(agent: AlgCgar3SeqMapfAgent):
    agent.path.append(agent.path[-1])


def all_execute_next_step(agents: Deque[AlgCgar3SeqMapfAgent], iteration: int) -> Dict[str, AlgCgar3SeqMapfAgent]:
    fs_to_a_dict: Dict[str, AlgCgar3SeqMapfAgent] = {}
    for agent in agents:
        agent.execute_simple_step(iteration)
        fs_to_a_dict[agent.curr_node.xy_name] = agent

    for a in agents:
        assert len(a.path) - 1 >= iteration
        assert a.path[iteration] == a.curr_node
    return fs_to_a_dict


def calc_pibt_step(
        main_agent: AlgCgar3SeqMapfAgent, agents: Deque[AlgCgar3SeqMapfAgent], nodes_dict: Dict[str, Node], h_dict: dict,
        given_goal_node: Node, blocked_nodes: List[Node], config_from: Dict[str, Node], config_to: Dict[str, Node],
        goals: Dict[str, Node], curr_n_name_to_agent_dict: Dict[str, AlgCgar3SeqMapfAgent],
        curr_n_name_to_agent_list: List[str], is_main_agent: bool = False,
        iteration: int = 0, to_assert: bool = False
) -> Dict[str, Node]:
    assert len(main_agent.path) == iteration
    # Preps
    for agent in agents:
        if agent.name not in config_to and len(agent.path) - 1 >= iteration:
            config_to[agent.name] = agent.path[iteration]
    if to_assert:
        assert len(set(config_to.values())) == len(set(config_to.keys()))

    # Calc PIBT
    curr_blocked_nodes = blocked_nodes[:]
    if is_main_agent:
        curr_blocked_nodes.append(given_goal_node)
    config_to = run_i_pibt(
        main_agent=main_agent, agents=agents, nodes_dict=nodes_dict, h_dict=h_dict,
        config_from=config_from, config_to=config_to, goals=goals,
        curr_n_name_to_agent_dict=curr_n_name_to_agent_dict, curr_n_name_to_agent_list=curr_n_name_to_agent_list,
        blocked_nodes=curr_blocked_nodes, given_goal_node=given_goal_node, iteration=iteration)
    # Update paths
    for agent in agents:
        if len(agent.path) - 1 == iteration - 1 and agent.name in config_to:
            next_node = config_to[agent.name]
            agent.path.append(next_node)

    return config_to


def calc_ep_steps(
        main_agent: AlgCgar3SeqMapfAgent, agents: Deque[AlgCgar3SeqMapfAgent],
        nodes: List[Node], nodes_dict: Dict[str, Node], h_dict: dict, given_goal_node: Node,
        curr_n_name_to_agent_dict: Dict[str, AlgCgar3SeqMapfAgent], curr_n_name_to_agent_list: List[str],
        a_non_sv_nodes_np: np.ndarray, iteration: int, to_assert: bool = False
) -> None:
    """
    - Build corridor
    - Build EP for ev-agents in the corridor
    - Evacuate ev-agents
    - Build the steps in the corridor to the main agent
    """
    assert len(main_agent.path) == iteration

    # Preps
    blocked_nodes = []
    # Calc
    corridor: List[Node] = build_corridor_from_nodes(main_agent.curr_node, given_goal_node, nodes_dict, h_dict, a_non_sv_nodes_np)
    # corridor: List[Node] = build_corridor(main_agent, nodes_dict, h_dict, a_non_sv_nodes_np, given_goal_node=given_goal_node)
    corridor_names: List[str] = [n.xy_name for n in corridor]
    assert corridor[0] == main_agent.path[-1]

    # if any of the corridor's nodes is blocked - just return
    if len([n for n in corridor[1:] if n in blocked_nodes]) > 0:
        return

    assert not (corridor[-1] == given_goal_node and corridor[-1].xy_name in curr_n_name_to_agent_dict)

    # Find ev-agents
    ev_agents: List[AlgCgar3SeqMapfAgent] = []
    for node in corridor[1:]:
        if node.xy_name in curr_n_name_to_agent_list:
            ev_agent = curr_n_name_to_agent_dict[node.xy_name]
            assert ev_agent != main_agent
            ev_agents.append(ev_agent)

    # Build ev-paths (evacuation paths) for ev-agents in the corridor
    ev_paths_list: List[List[Node]] = []
    captured_free_nodes: List[Node] = []
    blocked_nodes.extend([main_agent.curr_node, given_goal_node])
    blocked_nodes = list(set(blocked_nodes))
    for ev_agent in ev_agents:
        ev_path, captured_free_node = find_ev_path(
            ev_agent.curr_node, corridor, nodes_dict, blocked_nodes, captured_free_nodes,
            curr_n_name_to_agent_dict, curr_n_name_to_agent_list
        )
        if ev_path is None:
            return
        captured_free_nodes.append(captured_free_node)
        ev_paths_list.append(ev_path)

    # Build steps for the ev-agents inside the ev-paths + extend paths
    moved_agents = []
    last_visit_dict = {n.xy_name: 0 for n in nodes}
    for i_ev_path, ev_path in enumerate(ev_paths_list):
        curr_n_name_to_a_dict: Dict[str, AlgCgar3SeqMapfAgent] = {a.path[-1].xy_name: a for a in agents}
        curr_n_name_to_a_list: List[str] = list(curr_n_name_to_a_dict.keys())
        max_len, assigned_agents = push_ev_agents(ev_path, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                                                  moved_agents, nodes, main_agent, last_visit_dict,
                                                  iteration)
        assert main_agent not in assigned_agents
        # extend_other_paths(max_len, self.main_agent, self.agents)
        moved_agents.extend(assigned_agents)
        moved_agents = list(set(moved_agents))

    # Build the steps in the corridor to the main agent + extend the path
    push_main_agent(main_agent, corridor, moved_agents, iteration)


def update_agents_to_return(
        agents_to_return: List[AlgCgar3SeqMapfAgent], newly_planned_agents: List[AlgCgar3SeqMapfAgent], global_iteration: int
) -> List[AlgCgar3SeqMapfAgent]:
    to_add = False
    # if there are agents that were at their goals
    for p_agent in newly_planned_agents:
        if p_agent.curr_node == p_agent.get_goal_node():
            to_add = True
            break
    # if there are agents that touched paths of previously moved agents to return
    if not to_add:
        nodes_of_agents_to_return = []
        for r_agent in agents_to_return:
            nodes_of_agents_to_return.extend(r_agent.return_road_nodes)
        for p_agent in newly_planned_agents:
            new_plan = p_agent.path[global_iteration:]
            for new_n in new_plan:
                if new_n in nodes_of_agents_to_return:
                    to_add = True
                    break
    if to_add:
        for p_agent in newly_planned_agents:
            if p_agent not in agents_to_return:
                agents_to_return.append(p_agent)
    return agents_to_return


def get_future_captured_node_names(agents: Deque[AlgCgar3SeqMapfAgent], global_iteration: int) -> List[str]:
    future_captured_node_names: List[str] = []
    for agent in agents:
        if len(agent.path) - 1 > global_iteration:
            agent_path = agent.path[global_iteration:]
            # agent_path = agent.path[global_iteration + 1:]
            future_captured_node_names.extend([n.xy_name for n in agent_path])
    return future_captured_node_names


def all_update_return_roads(planned_agents: List[AlgCgar3SeqMapfAgent], iteration):
    # for the back-steps
    for agent in planned_agents:
        if len(agent.return_road) == 0:
            agent.return_road = deque([(agent.curr_node.xy_name, iteration-1, [], agent.curr_node)])
        agent_next_node = agent.path[iteration]
        if agent_next_node != agent.return_road[-1][3]:
            agent.return_road.append((agent_next_node.xy_name, iteration, [], agent_next_node))


def update_waiting_tables(
        agents_with_new_plan: List[AlgCgar3SeqMapfAgent], agents_to_return: List[AlgCgar3SeqMapfAgent],
        fs_to_a_dict: Dict[str, AlgCgar3SeqMapfAgent], to_assert: bool
) -> None:
    # update waiting_table
    for affected_agent in agents_to_return:
        affected_agent_name = affected_agent.name
        assert len(affected_agent.return_road) != 0
        if len(affected_agent.return_road) == 1:
            assert affected_agent.curr_node == affected_agent.return_road[-1][3]
            continue
        for n_name, i, a_list, n in affected_agent.return_road:
            # inside next_n_name_to_a_dict
            if n_name in fs_to_a_dict:
                agent_on_road = fs_to_a_dict[n_name]
                agent_on_road_name = agent_on_road.name
                if agent_on_road.curr_rank != 0 and agent_on_road != affected_agent:
                    assert agent_on_road in agents_with_new_plan
                    aor_n_name, aor_i, aor_a_list, aor_n = agent_on_road.return_road[-1]
                    assert aor_n_name == n_name
                    assert n == aor_n
                    aor_a_list.append(affected_agent.name)
                    affected_agent.add_to_wl(n, agent_on_road, aor_i, to_assert)


def calc_backward_road(
        backward_step_agents: List[AlgCgar3SeqMapfAgent],  # agents that are needed to be moved
        agents_with_new_plan: List[AlgCgar3SeqMapfAgent],  # agents that already planned
        agents_to_return: List[AlgCgar3SeqMapfAgent],  # all agents that need to return
        agents_dict: Dict[str, AlgCgar3SeqMapfAgent],
        from_n_to_a_dict: Dict[str, AlgCgar3SeqMapfAgent],
        future_captured_node_names: List[str],
        fs_to_a_dict: Dict[str, AlgCgar3SeqMapfAgent],
        to_config: Dict[str, Node],
        iteration: int, to_assert: bool = False
) -> None:
    # ------------------------------------------------------------------ #
    def update_data(given_a: AlgCgar3SeqMapfAgent, given_node: Node, to_pop: bool = False):
        to_config[given_a.name] = given_node
        fs_to_a_dict[given_node.xy_name] = given_a
        heapq.heappush(future_captured_node_names, given_node.xy_name)
        if to_pop:
            rr_n_name, rr_i, rr_a_list, rr_n = given_a.return_road.pop()
            assert given_node != rr_n
            assert given_node == given_a.return_road[-1][3]
            given_a.trash_return_road.append((rr_n_name, rr_i, rr_a_list, rr_n))
            item_to_remove = (given_a.name, rr_i)
            for rr_a_name in rr_a_list:
                rr_a = agents_dict[rr_a_name]
                rr_a.remove_from_wl(rr_n, given_a, rr_i)
    # ------------------------------------------------------------------ #
    # by this stage the forward_step_agents already executed their step
    if to_assert:
        for agent in backward_step_agents:
            assert agent.return_road[-1][3] == agent.curr_node
            assert agent.return_road[-1][3] == agent.path[-1]

    update_waiting_tables(agents_with_new_plan, agents_to_return, fs_to_a_dict, to_assert)

    # decide rest of to_config
    # open_list: Deque[AlgCgar3SeqMapfAgent] = deque(backward_step_agents[:])
    open_list: Deque[AlgCgar3SeqMapfAgent] = deque(agents_to_return[:])
    while len(open_list) > 0:
        next_agent = open_list.popleft()
        # already planned
        if next_agent.name in to_config:
            continue
        # no need to return, the agent wasn't displaced
        if len(next_agent.return_road) == 1:
            assert next_agent.return_road[0][3] == next_agent.curr_node
            update_data(next_agent, next_agent.curr_node)
            continue
        next_possible_n_name, next_rr_i, next_rr_a_list, next_possible_node = next_agent.return_road[-2]
        # next possible move is not allowed
        if next_possible_node.xy_name in future_captured_node_names:
            update_data(next_agent, next_agent.curr_node)
            continue
        # another agent in front of the agent needs to plan first
        # Circles: there will be never circles here => with the circles the algorithm will not work
        if next_possible_node.xy_name in from_n_to_a_dict:
            distur_agent = from_n_to_a_dict[next_possible_node.xy_name]
            assert distur_agent in agents_to_return
            if distur_agent.name not in to_config:
                open_list.append(next_agent)
                continue
        # no need to wait to anybody
        waiting_list = next_agent.get_wl(next_possible_node)
        if len(waiting_list) == 0:
            update_data(next_agent, next_possible_node, to_pop=True)
            continue
        last_captured_time = max([tpl[1] for tpl in waiting_list])
        if last_captured_time < next_rr_i:
            update_data(next_agent, next_possible_node, to_pop=True)
            continue
        # wait
        update_data(next_agent, next_agent.curr_node)
        continue

    # update a path
    for agent in backward_step_agents:
        assert len(agent.path) - 1 == iteration - 1
        to_node = to_config[agent.name]
        agent.path.append(to_node)
        if to_assert:
            assert agent.curr_node == agent.return_road[-1][3]
            assert agent.prev_node.xy_name in agent.curr_node.neighbours
    return


def clean_agents_to_return(
        agents_to_return: List[AlgCgar3SeqMapfAgent], iteration: int
) -> Tuple[List[AlgCgar3SeqMapfAgent], List[AlgCgar3SeqMapfAgent]]:
    cleaned_agents_to_return: List[AlgCgar3SeqMapfAgent] = []
    deleted_agents: List[AlgCgar3SeqMapfAgent] = []
    for agent in agents_to_return:
        # assert len(agent.path) - 1 >= iteration
        if len(agent.return_road) == 1 and len(agent.path[iteration:]) == 1:
            # assert agent.return_road[-1][3] == agent.path[iteration]
            deleted_agents.append(agent)
        else:
            cleaned_agents_to_return.append(agent)
    for da in deleted_agents:
        da.reset_return_road()
    return cleaned_agents_to_return, deleted_agents


def run_cgar(
        main_agent: AlgCgar3SeqMapfAgent,
        agents: Deque[AlgCgar3SeqMapfAgent],
        agents_dict: Dict[str, AlgCgar3SeqMapfAgent],
        non_sv_nodes_with_blocked_np: np.ndarray,
        h_dict: dict,
        img_np: np.ndarray,
        start_time: float,
        to_render: bool,
        to_assert: bool,
        alg_name: str,
        n_solved: int,
        n_agents: int,
        global_iteration: int
) -> Tuple[str, int]:
    if to_render:
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_rate = 0.001
        # plot_rate = 2

    # Preps
    agents_to_return: List[AlgCgar3SeqMapfAgent] = []
    finished_to_return: List[AlgCgar3SeqMapfAgent] = []
    main_goal_node = main_agent.get_goal_node()
    main_non_sv_nodes_np = non_sv_nodes_with_blocked_np[main_goal_node.x, main_goal_node.y]
    nodes: List[Node] = main_agent.nodes
    nodes_dict: Dict[str, Node] = main_agent.nodes_dict
    goals_dict: Dict[str, Node] = {a.name: a.get_goal_node() for a in agents}

    # if main_agent.num == 104:
    #     print()

    curr_loop = 0
    while True:
        curr_loop += 1

        # Preps
        next_global_iteration = global_iteration + 1
        config_from: Dict[str, Node] = {}
        config_to: Dict[str, Node] = {}
        curr_n_name_to_agent_dict: Dict[str, AlgCgar3SeqMapfAgent] = {}
        curr_n_name_to_agent_list: List[str] = []
        for agent in agents:
            config_from[agent.name] = agent.curr_node
            curr_n_name_to_agent_dict[agent.curr_node.xy_name] = agent
            assert agent.path[next_global_iteration - 1] == agent.curr_node
            heapq.heappush(curr_n_name_to_agent_list, agent.curr_node.xy_name)

        # CONTINUATION CHECK STAGE
        if main_agent.curr_node == main_agent.get_goal_node():
            if len(agents_to_return) > 0:
                stay_where_you_are(main_agent)
            else:
                if main_agent.alt_goal_node is not None:
                    changer_agent_name = main_agent.changer_agent_name
                    main_agent.remove_alt_goal_node()
                    if changer_agent_name == main_agent.name:
                        return 'start', global_iteration
                    else:
                        return 'end', global_iteration
                return 'out', global_iteration

        # CALC STEP STAGE
        assert main_agent == agents[0]
        unplanned_agents: List[AlgCgar3SeqMapfAgent] = [a for a in list(agents)[1:] if len(a.path) - 1 == global_iteration]
        if len(main_agent.path) - 1 == next_global_iteration - 1:
            a_next_node = get_min_h_nei_node(main_agent.curr_node, main_goal_node, nodes_dict, h_dict)
            if main_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
                # calc single PIBT step
                assert main_agent.curr_rank == 0
                blocked_nodes = [main_agent.get_goal_node()]
                calc_pibt_step(main_agent, agents, nodes_dict, h_dict, main_goal_node, blocked_nodes, config_from,
                               config_to, goals_dict, curr_n_name_to_agent_dict, curr_n_name_to_agent_list,
                               is_main_agent=True, iteration=next_global_iteration, to_assert=to_assert)
            else:
                # calc evacuation of agents from the corridor
                calc_ep_steps(main_agent, agents, nodes, nodes_dict, h_dict, main_goal_node,
                              curr_n_name_to_agent_dict, curr_n_name_to_agent_list, main_non_sv_nodes_np,
                              iteration=next_global_iteration, to_assert=to_assert)

        # RETURN STAGE
        newly_planned_agents: List[AlgCgar3SeqMapfAgent] = [a for a in unplanned_agents if len(a.path) - 1 > global_iteration]
        agents_to_return = update_agents_to_return(agents_to_return, newly_planned_agents, global_iteration)
        planned_agents = [a for a in agents_to_return if len(a.path) - 1 > global_iteration]
        backward_step_agents = [a for a in agents_to_return if len(a.path) - 1 == global_iteration]
        from_n_to_a_dict = curr_n_name_to_agent_dict
        future_captured_node_names = get_future_captured_node_names(agents, global_iteration)
        fs_to_a_dict = {a.path[next_global_iteration].xy_name: a for a in agents if len(a.path) - 1 >= next_global_iteration}
        to_config = {a.name: a.path[next_global_iteration] for a in agents if len(a.path) - 1 >= next_global_iteration}

        all_update_return_roads(planned_agents, next_global_iteration)
        calc_backward_road(
            backward_step_agents, planned_agents, agents_to_return, agents_dict, from_n_to_a_dict,
            future_captured_node_names, fs_to_a_dict, to_config, next_global_iteration, to_assert
        )
        agents_to_return, deleted_agents = clean_agents_to_return(agents_to_return, next_global_iteration)
        finished_to_return.extend(deleted_agents)
        print('', end='')

        # IF THERE IS NO PLAN THEN STAY
        for a in agents:
            if len(a.path) - 1 == next_global_iteration - 1:
                stay_where_you_are(a)

        fs_to_a_dict = all_execute_next_step(agents, next_global_iteration)
        global_iteration += 1

        # PRINT
        runtime = time.time() - start_time
        print(
            f'\r{'*' * 20} | [{alg_name}] | i: {global_iteration} | {main_agent.name} | solved: {n_solved}/{n_agents} | runtime: {runtime: .2f} seconds | {'*' * 20}',
            end='')
        # RENDER
        if to_render and global_iteration >= 0:
        # if to_render and n_solved >= 91:
            non_sv_nodes_np = non_sv_nodes_with_blocked_np[main_agent.get_goal_node().x, main_agent.get_goal_node().y]
            plot_info = {'img_np': img_np, 'agents': agents, 'i_agent': main_agent,
                         'non_sv_nodes_np': non_sv_nodes_np}
            plot_step_in_env(ax[0], plot_info)
            plot_return_paths(ax[1], plot_info)
            plt.pause(plot_rate)


def all_arrived(agents: Deque[AlgCgar3SeqMapfAgent], paths_dict: Dict[str, List[Node]]) -> bool:
    for agent in agents:
        if paths_dict[agent.name][-1] != agent.goal_node:
            return False
    return True


def update_chain_dict(
        chain_dict: Dict[str, str], config_from: Dict[str, Node], config_to: Dict[str, Node]
) -> Dict[str, str]:
    new_chain_dict: Dict[str, str] = {}
    for k, v in chain_dict.items():
        if k not in config_to:
            new_chain_dict[k] = v
    return new_chain_dict


def find_circles(chain_dict: Dict[str, str]) -> List[List[str]]:
    circles_list: List[List[str]] = []
    closed_list: List[str] = []
    open_list = list(chain_dict.keys())
    # chain_keys = list(chain_dict.keys())
    # heapq.heapify(chain_keys)
    while len(open_list) > 0:
        new_a_name = open_list.pop()
        if new_a_name in closed_list:
            continue
        new_circle_list = [new_a_name]
        next_a_name = chain_dict[new_a_name]
        is_circle = False

        while True:
            # if next_a_name not in chain_keys:
            if next_a_name not in chain_dict:
                break
            if next_a_name in closed_list:
                break
            if next_a_name in new_circle_list and next_a_name == new_circle_list[0]:
                circles_list.append(new_circle_list)
                is_circle = True
                break
            if next_a_name in new_circle_list and next_a_name != new_circle_list[0]:
                break
            new_circle_list.append(next_a_name)
            next_a_name = chain_dict[next_a_name]

        if is_circle:
            for new_c_name in new_circle_list:
                heapq.heappush(closed_list, new_c_name)
            # closed_list.extend(new_circle_list)
        else:
            # closed_list.append(new_a_name)
            heapq.heappush(closed_list, new_a_name)
    return circles_list


def resolve_circles(
        chain_dict: Dict[str, str], config_from: Dict[str, Node], config_to: Dict[str, Node],
        agents_roads_marks_dict: Dict[str, Deque[Tuple[str, int]]], waiting_table: Dict[str, Deque[Tuple[str, int]]],
        nodes_dict: Dict[str, Node], open_deq: Deque[AlgCgar3SeqMapfAgent], agents_dict: Dict[str, AlgCgar3SeqMapfAgent]
) -> bool:
    circles_list = find_circles(chain_dict)
    to_resume = True
    for circle in circles_list:
        to_resume = False
        for a_name in circle:
            a_road_marks = agents_roads_marks_dict[a_name]
            curr_n_name, curr_t = a_road_marks[0]
            curr_n_waiting_list = waiting_table[curr_n_name]
            next_possible_n_name, next_possible_t = a_road_marks[1]
            next_possible_n = nodes_dict[next_possible_n_name]
            config_to[a_name] = next_possible_n
            a_road_marks.popleft()
            curr_n_waiting_list.popleft()
            open_deq.remove(agents_dict[a_name])

    return to_resume


def compress_paths(
        agents: Deque[AlgCgar3SeqMapfAgent], agents_dict: Dict[str, AlgCgar3SeqMapfAgent],
        nodes: List[Node], nodes_dict: Dict[str, Node]
) -> Dict[str, List[Node]]:
    max_len = max([len(a.path) for a in agents])
    print()
    # ---------------------------------------------------------------------------------------------------------------- #
    print('\nStart to build road dicts...')
    agents_roads_dict: Dict[str, Deque[Node]] = {a.name: deque([a.start_node]) for a in agents}
    agents_roads_names_dict: Dict[str, Deque[str]] = {a.name: deque([a.start_node.xy_name]) for a in agents}
    agents_roads_marks_dict: Dict[str, Deque[Tuple[str, int]]] = {a.name: deque([(a.start_node.xy_name, 0)]) for a in agents}
    agents_roads_times_dict: Dict[str, Deque[Tuple[str, int]]] = {a.name: deque([(a.start_node.xy_name, 0)]) for a in agents}
    for agent in agents:
        agent_road = agents_roads_dict[agent.name]
        agent_road_names = agents_roads_names_dict[agent.name]
        agent_road_marks = agents_roads_marks_dict[agent.name]
        agent_road_times = agents_roads_times_dict[agent.name]
        for i, n in enumerate(agent.path[1:]):
            if n != agent_road[-1]:
                agent_road.append(n)
                agent_road_names.append(n.xy_name)
                agent_road_marks.append((n.xy_name, i))
                agent_road_times.append((n.xy_name, i))
            else:
                agent_road_times.append(agent_road_times[-1])
    print('Finished to build road dicts.')
    # ---------------------------------------------------------------------------------------------------------------- #
    print('Start to build waiting table...')
    waiting_table: Dict[str, Deque[Tuple[str, int]]] = {n.xy_name: deque() for n in nodes}
    for i in range(max_len):
        for agent in agents:
            agent_name = agent.name
            agent_road_times = agents_roads_times_dict[agent_name]
            curr_node_name, curr_first_arrive_time = agent_road_times[i]
            waiting_list = waiting_table[curr_node_name]
            if len(waiting_list) == 0:
                waiting_list.append((agent_name, curr_first_arrive_time))
            else:
                last_agent_name, last_agent_time = waiting_list[-1]
                if (last_agent_name, last_agent_time) == (agent_name, curr_first_arrive_time):
                    continue
                waiting_list.append((agent_name, curr_first_arrive_time))
    print('Finished to build waiting table.')
    # ---------------------------------------------------------------------------------------------------------------- #
    new_paths_dict: Dict[str, List[Node]] = {a.name: [a.start_node] for a in agents}
    iteration_next = 0
    while not all_arrived(agents, new_paths_dict):
        iteration_next += 1

        config_from: Dict[str, Node] = {a.name: new_paths_dict[a.name][iteration_next - 1] for a in agents}
        curr_n_name_to_a_dict = {config_from[a.name].xy_name: a for a in agents}
        curr_n_name_to_a_list = list(curr_n_name_to_a_dict.keys())
        heapq.heapify(curr_n_name_to_a_list)
        config_to: Dict[str, Node] = {}
        # config_to_n_names_list: List[str] = []
        chain_dict: Dict[str, str] = {}
        open_deq = deque(agents)

        iteration_config = 0
        while len(config_to) < len(config_from):
            iteration_config += 1
            if iteration_config > len(config_from) * 10:
                chain_dict = update_chain_dict(chain_dict, config_from, config_to)
                to_resume = resolve_circles(
                    chain_dict, config_from, config_to, agents_roads_marks_dict, waiting_table, nodes_dict,
                    open_deq, agents_dict
                )
                if not to_resume:
                    continue
            next_a = open_deq.popleft()
            next_a_name = next_a.name
            next_a_road_marks = agents_roads_marks_dict[next_a_name]

            assert next_a_name not in config_to

            curr_n_name, curr_t = next_a_road_marks[0]
            curr_n = nodes_dict[curr_n_name]
            curr_n_waiting_list = waiting_table[curr_n_name]
            curr_a_to_wait_name, curr_wait_time = curr_n_waiting_list[0]

            if len(next_a_road_marks) == 1:
                assert curr_n_name == next_a.goal_node.xy_name
                assert curr_a_to_wait_name == next_a_name
                config_to[next_a_name] = curr_n
                # check_configs(agents, config_from, config_to)
                continue

            next_possible_n_name, next_possible_t = next_a_road_marks[1]
            next_n_waiting_list = waiting_table[next_possible_n_name]
            first_a_to_wait_name, first_to_wait_name_time = next_n_waiting_list[0]

            if next_possible_n_name in curr_n_name_to_a_list:
                distur_a = curr_n_name_to_a_dict[next_possible_n_name]
                if distur_a.name not in config_to:
                    open_deq.append(next_a)
                    chain_dict[next_a_name] = distur_a.name
                    continue
                if distur_a.name in config_to and config_to[distur_a.name].xy_name == next_possible_n_name:
                    config_to[next_a_name] = curr_n
                    continue

            if (first_a_to_wait_name, first_to_wait_name_time) == (next_a_name, next_possible_t):
                assert (curr_a_to_wait_name, curr_wait_time) == (next_a_name, curr_t)
                next_possible_n = nodes_dict[next_possible_n_name]
                config_to[next_a_name] = next_possible_n
                next_a_road_marks.popleft()
                curr_n_waiting_list.popleft()
                continue
            else:
                config_to[next_a_name] = curr_n
                continue

        for a_name, node in config_to.items():
            new_paths_dict[a_name].append(node)

        solved: List[AlgCgar3SeqMapfAgent] = [a for a in agents if a.goal_node == new_paths_dict[a.name][-1]]
        print(f'\rCompressing paths... {iteration_next} | {len(solved)}/{len(agents)}', end='')
        # check_configs(agents, config_from, config_to, final_check=True)
    print('\nFinished to compress.')
    return new_paths_dict











