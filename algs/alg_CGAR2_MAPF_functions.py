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


class AlgCgar2MapfAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node, nodes: List[Node], nodes_dict: Dict[str, Node]):
        self.num = num
        self.priority = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.alt_goal_node: Node | None = None
        self.prev_goal_node_names_list: List[str] = []
        self.setting_agent_name: str | None = None
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
        assert self.prev_node.xy_name in self.curr_node.neighbours

    def get_goal_node(self) -> Node:
        if self.alt_goal_node is not None:
            return self.alt_goal_node
        return self.goal_node

    def reset_alt_goal_node(self, node: Node, setting_agent: Self) -> None:
        self.alt_goal_node = node
        self.setting_agent_name = setting_agent.name


    def remove_alt_goal_node(self) -> None:
        assert self.alt_goal_node is not None
        self.prev_goal_node_names_list.append(self.alt_goal_node.xy_name)
        self.alt_goal_node = None
        self.setting_agent_name = None

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


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# HELP FUNCTIONS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def stay_where_you_are(main_agent: AlgCgar2MapfAgent):
    main_agent.path.append(main_agent.path[-1])


def update_priority_numbers(agents: List[AlgCgar2MapfAgent]):
    for i_priority, agent in enumerate(agents):
        agent.priority = i_priority


def get_blocked_map(
    main_agent: AlgCgar2MapfAgent,
    agents: List[AlgCgar2MapfAgent],
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

    # Block return paths set by HR-agents
    for curr_priority, agent in enumerate(agents):
        if agent == main_agent:
            break
        for n in agent.return_road_nodes:
            blocked_map[n.x, n.y] = 1
        # Block original goal locations of HR-agents
        i_original_goal_node = agent.goal_node
        blocked_map[i_original_goal_node.x, i_original_goal_node.y] = 1
        # Block alt-goal locations set by HR-agents
        if agent.alt_goal_node is not None:
            i_alt_goal_node = agent.alt_goal_node
            blocked_map[i_alt_goal_node.x, i_alt_goal_node.y] = 1

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
        agent: AlgCgar2MapfAgent,
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
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# MAIN FUNCTIONS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def continuation_check_stage(
        main_agent: AlgCgar2MapfAgent,
        iteration: int,
        curr_n_name_to_a_dict: Dict[str, AlgCgar2MapfAgent],
        curr_n_name_to_a_list: List[str],
        goals_dict: Dict[str, Node],
        agents_to_return_dict: Dict[str, List[AlgCgar2MapfAgent]],
        agents: List[AlgCgar2MapfAgent],
        agents_dict: Dict[str, AlgCgar2MapfAgent],
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
    if len(main_agent.path) - 1 >= iteration:
        return True, {'message': 'the agent has a plan'}

    # If the agent is at its goal and has return paths to finish
    if main_agent.curr_node == main_agent.get_goal_node():
        main_a_return_agents_list = agents_to_return_dict[main_agent.name]
        if len(main_a_return_agents_list) > 0:
            stay_where_you_are(main_agent)
            return True, {'message': ''}
        # If the agent is at its goal and has no return paths to finish
        the_order_swapped = False
        if main_agent.alt_goal_node is not None:
            # Put back the previous order
            the_order_swapped = main_agent.setting_agent_name != main_agent.name
            if the_order_swapped:
                setting_agent = agents_dict[main_agent.setting_agent_name]
                prev_setting_agent_priority = setting_agent.priority
                setting_agent.priority = main_agent.priority
                main_agent.priority = prev_setting_agent_priority
                assert len(setting_agent.path) - 1 == iteration - 1
                stay_where_you_are(setting_agent)
            # Change the goal of the agent i back to the original
            main_agent.remove_alt_goal_node()
        stay_where_you_are(main_agent)
        message = 'swap back' if the_order_swapped else ''
        return False, {'message': message}

    # Create blocked map
    blocked_map: np.ndarray = get_blocked_map(main_agent, agents, img_np, iteration)

    # If the goal is blocked
    given_goal_node = main_agent.get_goal_node()
    if blocked_map[given_goal_node.x, given_goal_node.y] == 1:
        stay_where_you_are(main_agent)
        return True, {}

    blocked_nodes = [n for n in nodes if blocked_map[n.x, n.y] == 1]

    # (!) If the goal is not blocked and is occupied by someone
    if blocked_map[given_goal_node.x, given_goal_node.y] == 0 and given_goal_node.xy_name in curr_n_name_to_a_list:
        distur_agent = curr_n_name_to_a_dict[given_goal_node.xy_name]
        alter_goal_node = get_alter_goal_node(
            distur_agent, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_a_list,
            blocked_nodes, goals=list(goals_dict.values()), avoid_curr_nodes=True, avoid_goals=True)
        distur_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        if distur_agent.alt_goal_node is not None:
            assert distur_agent.priority > main_agent.priority
            prev_main_priority = main_agent.priority
            main_agent.priority = distur_agent.priority
            distur_agent.priority = prev_main_priority
        stay_where_you_are(distur_agent)
        stay_where_you_are(main_agent)
        assert len(agents_to_return_dict[distur_agent.name]) == 0
        assert len(agents_to_return_dict[main_agent.name]) == 0
        return False, {'message': 'swap'}

    main_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
    closest_corridor: List[Node] = build_corridor_from_nodes(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, main_non_sv_nodes_np
    )

    # If the next step/corridor is blocked somewhere
    if corridor_is_blocked_somewhere(closest_corridor, blocked_map):
        stay_where_you_are(main_agent)
        return True, {'message': ''}

    # (!) If the goal is unreachable and the reason is not in blocked nodes
    alt_is_good, alt_message, i_error, info = is_enough_free_locations(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_n_name_to_a_list, main_non_sv_nodes_np,
        blocked_nodes, full_corridor_check=True
    )
    if not alt_is_good and i_error == 5:
        alter_goal_node = get_alter_goal_node(
            main_agent, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_a_list,
            blocked_nodes, goals=list(goals_dict.values()), avoid_curr_nodes=True, avoid_goals=True)
        main_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        # continue to plan
        return True, {'message': ''}

    # if any other reason to fail
    if not alt_is_good:
        stay_where_you_are(main_agent)
        return True, {'message': f'{alt_message}'}

    return True, {'message': ''}


def calc_step_stage(
        main_agent: AlgCgar2MapfAgent,
        iteration: int,
        non_sv_nodes_with_blocked_np: np.ndarray,
) -> None:
    if len(main_agent.path) - 1 >= iteration:
        return
    # ---------------------------------------------------------------------------------------------------------- #
    # EXECUTE THE FORWARD STEP
    # ---------------------------------------------------------------------------------------------------------- #
    # decide on the goal
    given_goal_node = main_agent.get_goal_node()
    a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
    blocked_nodes = get_blocked_nodes_for_pibt(
        agents, iteration, need_to_freeze_main_goal_node, backward_step_agents
    )

    a_next_node = get_min_h_nei_node(agent.curr_node, given_goal_node, nodes_dict, h_dict)
    if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
        # calc single PIBT step
        # blocked_nodes = get_blocked_nodes(self.agents, iteration, self.need_to_freeze_main_goal_node)
        calc_pibt_step(agent, agents, nodes_dict, h_dict, given_goal_node, blocked_nodes, config_from, config_to,
                       goals, curr_n_name_to_agent_dict, curr_n_name_to_agent_list,
                       iteration=iteration, to_assert=to_assert)
        return 'plan of pibt'
    else:
        # calc evacuation of agents from the corridor
        calc_ep_steps(agent, agents, nodes, nodes_dict, h_dict, given_goal_node, config_from,
                      curr_n_name_to_agent_dict, curr_n_name_to_agent_list, a_non_sv_nodes_np,
                      backward_step_agents, need_to_freeze_main_goal_node, iteration, to_assert=to_assert)
        return 'plan of ev'


def return_agents_stage(
        main_agent: AlgCgar2MapfAgent
):
    pass










































