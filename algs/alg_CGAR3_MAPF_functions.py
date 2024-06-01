import heapq

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt
from algs.alg_CGAR import get_min_h_nei_node, build_corridor_from_nodes, find_ev_path, push_ev_agents, push_main_agent

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# AGENT CLASS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


class AlgCgar3MapfAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node, nodes: List[Node], nodes_dict: Dict[str, Node]):
        self.num = num
        self.curr_rank = num
        self.future_rank = num
        self.status = ''
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.alt_goal_node: Node | None = None
        self.prev_goal_node_names_list: List[str] = []
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.path: List[Node] = [start_node]
        self.return_road: Deque[Tuple[str, int, List[str], Node]] = deque()
        self.prev_return_daddies_names: List[str] = []
        self.trash_return_road: Deque[Tuple[str, int, List[str], Node]] = deque()
        # Dictionary: node.xy_name -> [ ( (0) agent name, (1) iteration ), ...]
        self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}

        self.parent_of_goal_node: Self = self
        self.parent_of_path: Self = self
        self.parent_of_return_road: Self | None = None

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
    def parent_of_return_road_name(self):
        return self.parent_of_return_road.name

    @property
    def parent_of_path_name(self):
        return self.parent_of_path.name

    @property
    def parent_of_goal_node_name(self):
        return self.parent_of_goal_node.name

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
        assert self.prev_node.xy_name in self.curr_node.neighbours

    def get_goal_node(self) -> Node:
        if self.alt_goal_node is not None:
            return self.alt_goal_node
        return self.goal_node

    def reset_alt_goal_node(self, node: Node, setting_agent: Self) -> None:
        self.alt_goal_node = node
        self.parent_of_goal_node = setting_agent

    def remove_alt_goal_node(self) -> None:
        assert self.alt_goal_node is not None
        self.prev_goal_node_names_list.append(self.alt_goal_node.xy_name)
        self.alt_goal_node = None
        self.parent_of_goal_node = self

    def remove_return_road(self):
        if self.parent_of_return_road is not None:
            if self.parent_of_return_road.num == 53 and self.num == 73:
                print('', end='')
        self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}
        self.return_road: Deque[Tuple[str, int, List[str], Node]] = deque()
        if self.parent_of_return_road is not None:
            self.prev_return_daddies_names.append(self.parent_of_return_road.name)
        self.parent_of_return_road = None

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


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# HELP FUNCTIONS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def update_status(agents: List[AlgCgar3MapfAgent]):
    for a in agents:
        a.status = ''


def update_ranks(agents: List[AlgCgar3MapfAgent]):
    for curr_rank, agent in enumerate(agents):
        agent.curr_rank = curr_rank
        agent.future_rank = curr_rank


def update_config_to(config_to: Dict[str, Node], agents: List[AlgCgar3MapfAgent], iteration: int):
    for agent in agents:
        if agent.name not in config_to and len(agent.path) - 1 >= iteration:
            config_to[agent.name] = agent.path[iteration]


def get_newly_planned_agents(
        unplanned_agents: List[AlgCgar3MapfAgent], config_to: Dict[str, Node], iteration: int
) -> List[AlgCgar3MapfAgent]:
    newly_planned_agents: List[AlgCgar3MapfAgent] = []
    for a in unplanned_agents:
        if len(a.path) - 1 >= iteration:
            newly_planned_agents.append(a)
            assert a.name in config_to
    return newly_planned_agents


def stay_where_you_are(main_agent: AlgCgar3MapfAgent, config_to: Dict[str, Node], iteration: int):
    # cut if needed
    if len(main_agent.path) - 1 >= iteration:
        config_to[main_agent.name] = main_agent.path[iteration]
        return
        # main_agent.path = main_agent.path[:iteration]
        # remove_return_paths_of_agent(main_agent)
    next_node: Node = main_agent.path[-1]
    assert next_node == main_agent.curr_node
    main_agent.path.append(next_node)
    config_to[main_agent.name] = next_node
    assert config_to[main_agent.name] == main_agent.path[iteration]


def set_parent_of_path(newly_planned_agents: List[AlgCgar3MapfAgent], parent: AlgCgar3MapfAgent) -> None:
    for a in newly_planned_agents:
        a.parent_of_path = parent


def get_blocked_nodes_from_map(nodes: List[Node], blocked_map: np.ndarray, do_not_block_curr_nodes: bool = False) -> List[Node]:
    blocked_nodes = []
    for n in nodes:
        if blocked_map[n.x, n.y] == 1:
            blocked_nodes.append(n)
    if do_not_block_curr_nodes:
        # for agent in agents:
        #     if len(agent.path) - 1 >= iteration:
        #         # future_path = agent.path[iteration - 1:]
        #         curr_n = agent.path[iteration - 1]
        #         blocked_map[curr_n.x, curr_n.y] = 0
        pass
    return blocked_nodes


def init_blocked_map(
    agents: List[AlgCgar3MapfAgent],
    img_np: np.ndarray,
    iteration: int,
) -> np.ndarray:
    blocked_map: np.ndarray = np.zeros(img_np.shape)
    # Block all future steps of everyone
    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            future_path = agent.path[iteration - 1:]
            for n in future_path:
                blocked_map[n.x, n.y] = 1
    return blocked_map


def update_blocked_map(
    blocked_map: np.ndarray,
    main_agent: AlgCgar3MapfAgent,
    hr_agents: List[AlgCgar3MapfAgent],
    newly_planned_agents: List[AlgCgar3MapfAgent],
    agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
    iteration: int,
) -> np.ndarray:

    for agent in newly_planned_agents:
        future_path = agent.path[iteration - 1:]
        for n in future_path:
            blocked_map[n.x, n.y] = 1

    # Block return paths set by HR-agents
    for n in main_agent.return_road_nodes:
        blocked_map[n.x, n.y] = 1
    hr_return_agents = agents_to_return_dict[main_agent.name]
    for sub_a in hr_return_agents:
        for n in sub_a.return_road_nodes:
            blocked_map[n.x, n.y] = 1
    return blocked_map


def get_blocked_map(
    main_agent: AlgCgar3MapfAgent,
    hr_agents: List[AlgCgar3MapfAgent],
    lr_agents: List[AlgCgar3MapfAgent],
    agents: List[AlgCgar3MapfAgent],
    agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
    img_np: np.ndarray,
    iteration: int,
) -> np.ndarray:
    """
    Block all future steps of everyone
    Block return paths set by HR-agents
    Block goal locations of HR-agents
    Block alt-goal locations set by HR-agents
    """
    blocked_map: np.ndarray = np.zeros(img_np.shape)

    # Block all future steps of everyone
    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            future_path = agent.path[iteration - 1:]
            for n in future_path:
                blocked_map[n.x, n.y] = 1

            # if blocked_map[27, 9]:
            #     print()

    # Block return paths set by HR-agents
    for curr_priority, hr_agent in enumerate(hr_agents):

        for n in hr_agent.return_road_nodes:
            blocked_map[n.x, n.y] = 1

            # if blocked_map[27, 9]:
            #     print()

        hr_return_agents = agents_to_return_dict[hr_agent.name]
        for sub_a in hr_return_agents:
            for n in sub_a.return_road_nodes:
                blocked_map[n.x, n.y] = 1

                # if blocked_map[27, 9]:
                #     print()

        # # Block original goal locations of HR-agents
        # i_original_goal_node = hr_agent.goal_node
        # blocked_map[i_original_goal_node.x, i_original_goal_node.y] = 1
        #
        # # Block alt-goal locations set by HR-agents
        # if hr_agent.alt_goal_node is not None:
        #     i_alt_goal_node = hr_agent.alt_goal_node
        #     blocked_map[i_alt_goal_node.x, i_alt_goal_node.y] = 1

        # i_goal_node = hr_agent.get_goal_node()
        # blocked_map[i_goal_node.x, i_goal_node.y] = 1

    return blocked_map


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
        agent: AlgCgar3MapfAgent,
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        non_sv_nodes_with_blocked_np: np.ndarray,
        curr_n_name_to_agent_list: List[str],
        blocked_nodes: List[Node],
        goals: List[Node] | None = None,
        avoid_curr_nodes: bool = False,
        avoid_goals: bool = False,
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

        alt_is_good = False
        if not_curr_node and non_sv_in_main and not_in_curr_nodes and not_in_goal_nodes:
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


def corridor_is_blocked_somewhere(closest_corridor: List[Node], blocked_map: np.ndarray) -> bool:
    for n in closest_corridor:
        if blocked_map[n.x, n.y] == 1:
            return True
    return False


def calc_pibt_step(
        main_agent: AlgCgar3MapfAgent, agents: List[AlgCgar3MapfAgent], nodes_dict: Dict[str, Node], h_dict: dict,
        given_goal_node: Node, blocked_nodes: List[Node], config_from: Dict[str, Node], config_to: Dict[str, Node],
        goals: Dict[str, Node], curr_n_name_to_agent_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_agent_list: List[str], is_main_agent: bool = False,
        iteration: int = 0, to_assert: bool = False
) -> Dict[str, Node]:
    # print(f'\n --- inside calc_pibt_step {iteration} --- ')
    assert len(main_agent.path) == iteration
    # if one_of_best_next_nodes_are_blocked(main_agent, given_goal_node, blocked_nodes, nodes_dict, h_dict):
    #     next_node = main_agent.path[-1]
    #     config_to[main_agent.name] = next_node
    #     main_agent.path.append(next_node)
    #     return config_to
    # Preps
    # config_to = {}
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
        main_agent: AlgCgar3MapfAgent, agents: List[AlgCgar3MapfAgent], nodes: List[Node], nodes_dict: Dict[str, Node],
        h_dict: dict, given_goal_node: Node, config_from: Dict[str, Node],
        curr_n_name_to_agent_dict: Dict[str, AlgCgar3MapfAgent], curr_n_name_to_agent_list: List[str],
        a_non_sv_nodes_np: np.ndarray, blocked_nodes: List[Node], iteration: int
) -> None:
    """
    - Build corridor
    - Build EP for ev-agents in the corridor
    - Evacuate ev-agents
    - Build the steps in the corridor to the main agent
    """
    assert len(main_agent.path) == iteration

    # Preps
    blocked_nodes_names: List[str] = [n.xy_name for n in blocked_nodes]
    # Calc
    corridor: List[Node] = build_corridor_from_nodes(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, a_non_sv_nodes_np
    )
    corridor_names: List[str] = [n.xy_name for n in corridor]
    assert corridor[0] == main_agent.path[-1]

    # if any of the corridor's nodes is blocked - just return
    if len([n for n in corridor[1:] if n in blocked_nodes]) > 0:
        return

    assert not (corridor[-1] == given_goal_node and corridor[-1].xy_name in curr_n_name_to_agent_dict)

    # Find ev-agents
    ev_agents: List[AlgCgar3MapfAgent] = []
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
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent] = {a.path[-1].xy_name: a for a in agents}
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


def remove_return_paths_of_agent(
        agent_daddy: AlgCgar3MapfAgent, agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]]
) -> None:
    if agent_daddy is None:
        return
    prev_daddy_return_agents = agents_to_return_dict[agent_daddy.name]
    assert agent_daddy not in prev_daddy_return_agents
    # if len(prev_daddy_return_agents) > 0:
    #     assert len(agent_daddy.return_road) == 0
    for a in prev_daddy_return_agents:
        a.remove_return_road()
    agents_to_return_dict[agent_daddy.name] = []


def update_agents_to_return(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
        agents_to_return: List[AlgCgar3MapfAgent],
        newly_planned_agents: List[AlgCgar3MapfAgent],
        iteration: int
) -> List[AlgCgar3MapfAgent]:
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
            new_plan = p_agent.path[iteration - 1:]
            for new_n in new_plan:
                if new_n in nodes_of_agents_to_return:
                    to_add = True
                    break
    if to_add:
        for p_agent in newly_planned_agents:
            # assert p_agent in lr_agents
            if p_agent not in agents_to_return:
                # Delete all return paths of other LR-agents that are related to the newly moved agents
                if p_agent.parent_of_return_road is not None:
                    prev_return_daddy = p_agent.parent_of_return_road
                    # assert prev_return_daddy in lr_agents
                    remove_return_paths_of_agent(prev_return_daddy, agents_to_return_dict)
                p_agent.parent_of_return_road = main_agent
                agents_to_return.append(p_agent)
    return agents_to_return


def update_future_captured_node_names(future_captured_node_names: List[str], agents: List[AlgCgar3MapfAgent], iteration: int) -> List[str]:
    for agent in agents:
        agent_path = agent.path[iteration - 1:]
        # agent_path = agent.path[global_iteration + 1:]
        for n in agent_path:
            if n.xy_name not in future_captured_node_names:
                heapq.heappush(future_captured_node_names, n.xy_name)
    return future_captured_node_names


def get_future_captured_node_names(agents: List[AlgCgar3MapfAgent], iteration: int) -> List[str]:
    future_captured_node_names: List[str] = []
    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            agent_path = agent.path[iteration - 1:]
            # agent_path = agent.path[global_iteration + 1:]
            future_captured_node_names.extend([n.xy_name for n in agent_path])
    return future_captured_node_names


def all_update_return_roads(planned_agents: List[AlgCgar3MapfAgent], iteration):
    # for the back-steps
    for agent in planned_agents:
        if len(agent.return_road) == 0:
            agent.return_road = deque([(agent.curr_node.xy_name, iteration-1, [], agent.curr_node)])
        agent_next_node = agent.path[iteration]
        if agent_next_node != agent.return_road[-1][3]:
            agent.return_road.append((agent_next_node.xy_name, iteration, [], agent_next_node))


def update_waiting_tables(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        planned_agents: List[AlgCgar3MapfAgent],
        agents_to_return: List[AlgCgar3MapfAgent],
        fs_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        future_captured_node_names: List[str],
        iteration: int, to_assert: bool
) -> None:
    # update waiting_table
    for affected_agent in agents_to_return:
        affected_agent_name = affected_agent.name
        assert len(affected_agent.return_road) != 0
        if len(affected_agent.return_road) == 1:
            assert affected_agent.curr_node == affected_agent.return_road[-1][3]
            continue
        for n_name, i, a_list, n in affected_agent.return_road:
            # inside fs_to_a_dict
            if n_name in fs_to_a_dict:
                agent_on_road = fs_to_a_dict[n_name]
                agent_on_road_name = agent_on_road.name
                if agent_on_road != affected_agent and agent_on_road != main_agent and agent_on_road not in hr_agents:
                    assert agent_on_road in planned_agents
                    aor_n_name, aor_i, aor_a_list, aor_n = agent_on_road.return_road[-1]
                    assert aor_n_name == n_name
                    assert n == aor_n
                    aor_a_list.append(affected_agent.name)
                    affected_agent.add_to_wl(n, agent_on_road, aor_i, to_assert)


def calc_backward_road(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        backward_step_agents: List[AlgCgar3MapfAgent],  # agents that are needed to be moved
        planned_agents: List[AlgCgar3MapfAgent],  # agents that already planned
        agents_to_return: List[AlgCgar3MapfAgent],  # all agents that need to return
        agents_dict: Dict[str, AlgCgar3MapfAgent],
        from_n_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        future_captured_node_names: List[str],
        fs_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        to_config: Dict[str, Node],
        iteration: int, to_assert: bool = False
) -> None:
    # ------------------------------------------------------------------ #
    def update_data(given_a: AlgCgar3MapfAgent, given_node: Node, to_pop: bool = False):
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

    update_waiting_tables(
        main_agent, hr_agents, lr_agents, planned_agents, agents_to_return, fs_to_a_dict, future_captured_node_names,
        iteration, to_assert
    )

    # decide rest of to_config
    # open_list: Deque[AlgCgar3MapfAgent] = deque(backward_step_agents[:])
    open_list: Deque[AlgCgar3MapfAgent] = deque(agents_to_return[:])
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
        if next_possible_n_name in future_captured_node_names:
            update_data(next_agent, next_agent.curr_node)
            continue
        # another agent in front of the agent needs to plan first
        # Circles: there will be never circles here => with the circles the algorithm will not work
        if next_possible_n_name in from_n_to_a_dict:
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
        agents_to_return: List[AlgCgar3MapfAgent], iteration: int
) -> Tuple[List[AlgCgar3MapfAgent], List[AlgCgar3MapfAgent]]:
    cleaned_agents_to_return: List[AlgCgar3MapfAgent] = []
    deleted_agents: List[AlgCgar3MapfAgent] = []
    for agent in agents_to_return:
        # assert len(agent.path) - 1 >= iteration
        if len(agent.return_road) == 1 and len(agent.path[iteration:]) == 1:
            # assert agent.return_road[-1][3] == agent.path[iteration]
            deleted_agents.append(agent)
        else:
            cleaned_agents_to_return.append(agent)
    for da in deleted_agents:
        da.remove_return_road()
    return cleaned_agents_to_return, deleted_agents


def remove_crossed_return_paths_of_lr_agents(
        main_agent: AlgCgar3MapfAgent, lr_agents: List[AlgCgar3MapfAgent],
        iteration: int, agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
) -> None:
    future_main_path_set = set(main_agent.path[iteration-1:])
    for lr_agent in lr_agents:
        lr_return_agents = agents_to_return_dict[lr_agent.name]
        for lr_return_agent in lr_return_agents:
            if len(future_main_path_set.intersection(lr_return_agent.return_road_nodes)) != 0:
                remove_return_paths_of_agent(lr_agent, agents_to_return_dict)
                break


def if_set_by_hr_a_cross_me_remove_my_return_paths(
        main_agent: AlgCgar3MapfAgent, hr_agents: List[AlgCgar3MapfAgent],
        newly_planned_agents: List[AlgCgar3MapfAgent], agents: List[AlgCgar3MapfAgent],
        iteration: int, agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
config_to: Dict[str, Node], agents_dict: Dict[str, AlgCgar3MapfAgent],
) -> None:
    main_agents_to_return = agents_to_return_dict[main_agent.name]
    agents_to_return_nodes_names: List[str] = []
    for agent_to_return in main_agents_to_return:
        # assert len(agent_to_return.return_road) > 0
        for n in agent_to_return.return_road_nodes[-2:]:
            heapq.heappush(agents_to_return_nodes_names, n.xy_name)

    to_remove_return_paths = False
    for agent_name, config_n in config_to.items():
        agent = agents_dict[agent_name]
        if agent.parent_of_path.curr_rank >= main_agent.curr_rank:
            continue
        if config_n.xy_name in agents_to_return_nodes_names:
            to_remove_return_paths = True
            break
        # for n in agent.path[iteration - 1:]:
        #     if n.xy_name in agents_to_return_nodes_names:
        #         to_remove_return_paths = True
        #         break
        # if to_remove_return_paths:
        #     break

    # to_remove_return_paths = False
    # for agent in agents:
    #     if agent.parent_of_path.curr_rank >= main_agent.curr_rank:
    #         continue
    #     if len(agent.path) - 1 == iteration - 1:
    #         continue
    #     for n in agent.path[iteration - 1:]:
    #         if n.xy_name in agents_to_return_nodes_names:
    #             to_remove_return_paths = True
    #             break
    #     if to_remove_return_paths:
    #         break

    if to_remove_return_paths:
        remove_return_paths_of_agent(main_agent, agents_to_return_dict)
        remove_return_paths_of_agent(main_agent.parent_of_return_road, agents_to_return_dict)


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# MAIN FUNCTIONS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def continuation_check_stage(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        blocked_map: np.ndarray,
        iteration: int,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals_dict: Dict[str, Node],
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_a_list: List[str],
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
        agents: List[AlgCgar3MapfAgent],
        agents_dict: Dict[str, AlgCgar3MapfAgent],
        img_np: np.ndarray,
        h_dict: dict,
        non_sv_nodes_with_blocked_np: np.ndarray,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
) -> Tuple[bool, dict]:
    """
    returns: to_resume: bool, info: dict
    """

    # if the agent has a plan
    if main_agent.name in config_to:
        return True, {'message': 'in config_to'}

    # If the agent is at its goal and has return paths to finish
    if main_agent.curr_node == main_agent.get_goal_node():
        main_a_return_agents_list = agents_to_return_dict[main_agent.name]
        if len(main_a_return_agents_list) > 0:
            stay_where_you_are(main_agent, config_to, iteration)
            return True, {'message': 'is at the goal, but need to wait for agents to return'}
        # If the agent is at its goal and has no return paths to finish
        the_order_swapped = False
        if main_agent.alt_goal_node is not None:
            # Put back the previous order
            parent = main_agent.parent_of_goal_node
            the_order_swapped = parent != main_agent
            if the_order_swapped:
                if parent.future_rank > main_agent.future_rank:
                    prev_main_rank = main_agent.future_rank
                    main_agent.future_rank = parent.future_rank
                    parent.future_rank = prev_main_rank
                    remove_return_paths_of_agent(parent, agents_to_return_dict)
                    remove_return_paths_of_agent(parent.parent_of_return_road, agents_to_return_dict)
                stay_where_you_are(parent, config_to, iteration)
            # Change the goal of the agent i back to the original
            main_agent.remove_alt_goal_node()
        stay_where_you_are(main_agent, config_to, iteration)
        message = 'swap back' if the_order_swapped else ''
        return False, {'message': message}

    # Create blocked map
    # blocked_map: np.ndarray = get_blocked_map(main_agent, hr_agents, lr_agents, agents, agents_to_return_dict, img_np, iteration)
    given_goal_node = main_agent.get_goal_node()
    main_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
    main_next_node = get_min_h_nei_node(main_agent.curr_node, given_goal_node, nodes_dict, h_dict)
    closest_corridor: List[Node] = build_corridor_from_nodes(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, main_non_sv_nodes_np
    )

    if main_non_sv_nodes_np[main_next_node.x, main_next_node.y] == 1:
        # PIBT step
        if main_next_node != given_goal_node:
            return True, {'message': 'pibt step'}
    else:
        # EP step
        # If the next step/corridor is blocked somewhere
        if corridor_is_blocked_somewhere(closest_corridor, blocked_map):
            stay_where_you_are(main_agent, config_to, iteration)
            return True, {'message': 'the next step/corridor is blocked somewhere'}

    blocked_nodes = [n for n in nodes if blocked_map[n.x, n.y] == 1]

    # (!) If the goal is in next step/corridor and is not blocked and is occupied by someone
    goal_in_corridor = given_goal_node in closest_corridor
    goal_not_blocked = blocked_map[given_goal_node.x, given_goal_node.y] == 0
    goal_is_occupied = given_goal_node.xy_name in curr_n_name_to_a_list
    if goal_in_corridor and goal_not_blocked and goal_is_occupied:
        distur_agent = curr_n_name_to_a_dict[given_goal_node.xy_name]
        alter_goal_node = get_alter_goal_node(
            distur_agent, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_a_list,
            blocked_nodes, goals=list(goals_dict.values()), avoid_curr_nodes=True, avoid_goals=True)
        distur_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        if distur_agent.alt_goal_node is not None:
            # assert distur_agent in lr_agents
            if distur_agent.future_rank > main_agent.future_rank:
                prev_main_priority = main_agent.future_rank
                main_agent.future_rank = distur_agent.future_rank
                distur_agent.future_rank = prev_main_priority
        stay_where_you_are(distur_agent, config_to, iteration)
        stay_where_you_are(main_agent, config_to, iteration)

        remove_return_paths_of_agent(distur_agent, agents_to_return_dict)
        remove_return_paths_of_agent(distur_agent.parent_of_return_road, agents_to_return_dict)
        remove_return_paths_of_agent(main_agent, agents_to_return_dict)
        remove_return_paths_of_agent(main_agent.parent_of_return_road, agents_to_return_dict)

        return True, {'message': 'swap'}

    alt_is_good, alt_message, i_error, info = is_enough_free_locations(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_n_name_to_a_list, main_non_sv_nodes_np,
        blocked_nodes, full_corridor_check=True
    )
    # (!) If the goal is unreachable and the reason is not in blocked nodes
    if not alt_is_good and i_error == 5:
        alter_goal_node = get_alter_goal_node(
            main_agent, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_a_list,
            blocked_nodes, goals=list(goals_dict.values()), avoid_curr_nodes=True, avoid_goals=True)
        if alter_goal_node == main_agent.goal_node:
            stay_where_you_are(main_agent, config_to, iteration)
            remove_return_paths_of_agent(main_agent, agents_to_return_dict)
            remove_return_paths_of_agent(main_agent.parent_of_return_road, agents_to_return_dict)
            return True, {'message': 'alt goal node is not good'}
        main_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        remove_return_paths_of_agent(main_agent, agents_to_return_dict)
        remove_return_paths_of_agent(main_agent.parent_of_return_road, agents_to_return_dict)
        return True, {'message': 'error 5'}

    # if any other reason to fail
    if not alt_is_good:
        stay_where_you_are(main_agent, config_to, iteration)
        remove_return_paths_of_agent(main_agent, agents_to_return_dict)
        remove_return_paths_of_agent(main_agent.parent_of_return_road, agents_to_return_dict)
        return True, {'message': f'{alt_message}'}

    return True, {'message': 'continue'}


def calc_step_stage(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        blocked_map: np.ndarray,
        iteration: int,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals_dict: Dict[str, Node],
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_a_list: List[str],
        non_sv_nodes_with_blocked_np: np.ndarray,
        agents: List[AlgCgar3MapfAgent],
        agents_dict: Dict[str, AlgCgar3MapfAgent],
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        img_np: np.ndarray,
        h_dict: dict,
) -> str:
    main_agent.status = 'was main_agent'
    if main_agent.name in config_to:
        message = 'already planned'
        main_agent.last_calc_step_message = message
        return message
    # ---------------------------------------------------------------------------------------------------------- #
    # EXECUTE THE FORWARD STEP
    # ---------------------------------------------------------------------------------------------------------- #
    # decide on the goal
    given_goal_node = main_agent.get_goal_node()
    a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
    # blocked_map: np.ndarray = get_blocked_map(main_agent, hr_agents, lr_agents, agents, agents_to_return_dict, img_np, iteration)
    blocked_nodes = get_blocked_nodes_from_map(nodes, blocked_map)

    a_next_node = get_min_h_nei_node(main_agent.curr_node, given_goal_node, nodes_dict, h_dict)
    if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
        # calc single PIBT step
        # blocked_nodes = get_blocked_nodes(self.agents, iteration, self.need_to_freeze_main_goal_node)
        calc_pibt_step(main_agent, agents, nodes_dict, h_dict, given_goal_node, blocked_nodes, config_from, config_to,
                       goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                       iteration=iteration)
        message = f'plan of pibt in i {iteration}'
    else:
        # calc evacuation of agents from the corridor
        calc_ep_steps(main_agent, agents, nodes, nodes_dict, h_dict, given_goal_node, config_from,
                      curr_n_name_to_a_dict, curr_n_name_to_a_list, a_non_sv_nodes_np,
                      blocked_nodes, iteration)
        message = f'plan of ev in i {iteration}'

    update_config_to(config_to, agents, iteration)

    main_agent.last_calc_step_message = message
    return message


def return_agents_stage(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        iteration: int,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals_dict: Dict[str, Node],
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_a_list: List[str],
        newly_planned_agents: List[AlgCgar3MapfAgent],
        future_captured_node_names: List[str],
        agents: List[AlgCgar3MapfAgent],
        agents_dict: Dict[str, AlgCgar3MapfAgent],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
) -> None:

    if_set_by_hr_a_cross_me_remove_my_return_paths(
        main_agent, hr_agents, newly_planned_agents, agents, iteration, agents_to_return_dict,
        config_to, agents_dict
    )

    agents_to_return = agents_to_return_dict[main_agent.name]

    agents_to_return = update_agents_to_return(
        main_agent, hr_agents, lr_agents, agents_to_return_dict, agents_to_return, newly_planned_agents, iteration
    )

    agents_to_return_dict[main_agent.name] = agents_to_return
    assert main_agent not in agents_to_return

    planned_agents = [a for a in agents_to_return if len(a.path) - 1 >= iteration]
    backward_step_agents = [a for a in agents_to_return if len(a.path) - 1 == iteration - 1]
    from_n_to_a_dict = curr_n_name_to_a_dict
    fs_to_a_dict = {node.xy_name: agents_dict[agent_name] for agent_name, node in config_to.items()}

    all_update_return_roads(planned_agents, iteration)

    calc_backward_road(
        main_agent, hr_agents, lr_agents, backward_step_agents, planned_agents, agents_to_return, agents_dict, from_n_to_a_dict,
        future_captured_node_names, fs_to_a_dict, config_to, iteration,
    )

    agents_to_return, deleted_agents = clean_agents_to_return(agents_to_return, iteration)
    agents_to_return_dict[main_agent.name] = agents_to_return

    # update_config_to(config_to, agents, iteration)
    update_config_to(config_to, backward_step_agents, iteration)

    return





















