import heapq

import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
# from algs.alg_PIBT import run_i_pibt
from algs.alg_CGAR import get_min_h_nei_node, build_corridor_from_nodes, find_ev_path_m, push_ev_agents, push_main_agent

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
        self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}
        self.return_road: Deque[Tuple[str, int, List[str], Node]] = deque()
        # assert self.parent_of_path != self
        self.prev_return_daddies_names.append(self.parent_of_path_name)

    def set_parent_of_path(self, parent):
        self.parent_of_path = parent

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
    next_node: Node = main_agent.path[-1]
    assert next_node == main_agent.curr_node
    main_agent.path.append(next_node)
    config_to[main_agent.name] = next_node
    assert config_to[main_agent.name] == main_agent.path[iteration]
    main_agent.set_parent_of_path(main_agent)


def update_parent_of_path(newly_planned_agents: List[AlgCgar3MapfAgent], parent: AlgCgar3MapfAgent) -> None:
    for a in newly_planned_agents:
        a.set_parent_of_path(parent)


def get_blocked_nodes_from_map(nodes: List[Node], nodes_dict: Dict[str, Node], blocked_map: np.ndarray, do_not_block_curr_nodes: bool = False) -> List[Node]:
    blocked_nodes = []
    indices = np.where(blocked_map == 1)
    for x, y in zip(indices[0], indices[1]):
        n = nodes_dict[f'{x}_{y}']
        blocked_nodes.append(n)

    # for x in range(blocked_map.shape[0]):
    #     for y in range(blocked_map.shape[1]):
    #         if blocked_map[x, y] == 1:
    #             n = nodes_dict[f'{x}_{y}']
    #             blocked_nodes.append(n)

    # for n in nodes:
    #     if blocked_map[n.x, n.y] == 1:
    #         blocked_nodes.append(n)
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
) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    blocked_map: np.ndarray = np.zeros(img_np.shape)
    r_blocked_map: np.ndarray = np.zeros(img_np.shape)
    future_captured_node_names: List[str] = []
    # Block all future steps of everyone
    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            future_path = agent.path[iteration - 1:]
            for n in future_path:
                blocked_map[n.x, n.y] = 1
                heapq.heappush(future_captured_node_names, n.xy_name)
        else:
            n = agent.path[iteration - 1]
            heapq.heappush(future_captured_node_names, n.xy_name)
    return blocked_map, r_blocked_map, future_captured_node_names


def update_blocked_map(
    blocked_map: np.ndarray,
    r_blocked_map: np.ndarray,
    main_agent: AlgCgar3MapfAgent,
    hr_agents: List[AlgCgar3MapfAgent],
    newly_planned_agents: List[AlgCgar3MapfAgent],
    agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
    iteration: int,
) -> Tuple[np.ndarray, np.ndarray]:

    for agent in newly_planned_agents:
        future_path = agent.path[iteration - 1:]
        for n in future_path:
            blocked_map[n.x, n.y] = 1

    # Block return paths set by HR-agents
    # return road of HR
    for n in main_agent.return_road_nodes:
        blocked_map[n.x, n.y] = 1
        r_blocked_map[n.x, n.y] = 1
    # return roads set by HR
    hr_return_agents = agents_to_return_dict[main_agent.name]
    for sub_a in hr_return_agents:
        for n in sub_a.return_road_nodes:
            blocked_map[n.x, n.y] = 1
            r_blocked_map[n.x, n.y] = 1

    # goal node set by HR
    if main_agent.curr_rank == 0:
        main_goal_node = main_agent.get_goal_node()
        blocked_map[main_goal_node.x, main_goal_node.y] = 1
        r_blocked_map[main_goal_node.x, main_goal_node.y] = 1

    return blocked_map, r_blocked_map


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

    # Block return paths set by HR-agents
    for curr_priority, hr_agent in enumerate(hr_agents):

        for n in hr_agent.return_road_nodes:
            blocked_map[n.x, n.y] = 1

        hr_return_agents = agents_to_return_dict[hr_agent.name]
        for sub_a in hr_return_agents:
            for n in sub_a.return_road_nodes:
                blocked_map[n.x, n.y] = 1

    return blocked_map


def is_enough_free_locations(
        curr_node: Node,
        goal_node: Node,
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        curr_n_name_to_agent_list: List[str],
        non_sv_nodes_np: np.ndarray,
        blocked_map: np.ndarray | None = None,
        full_corridor_check: bool = False
) -> Tuple[bool, str, int, dict]:
    next_node = get_min_h_nei_node(curr_node, goal_node, nodes_dict, h_dict)
    full_path: List[Node] = [next_node]
    open_list: List[Node] = [next_node]
    open_list_names: List[str] = [next_node.xy_name]
    closed_list: List[Node] = [curr_node, goal_node]
    closed_list_names: List[str] = [n.xy_name for n in closed_list]
    heapq.heapify(closed_list_names)
    # blocked_nodes_names = [n.xy_name for n in blocked_nodes]
    # heapq.heapify(blocked_nodes_names)

    if next_node == goal_node:
        # if next_node in other_curr_nodes or next_node in blocked_nodes:
        if next_node.xy_name in curr_n_name_to_agent_list or blocked_map[next_node.x, next_node.y]:
            return False, f'PROBLEM-1 - next node {next_node.xy_name} is a goal node and is occupied or blocked', 1, {
                'goal_node': goal_node.xy_name,
                'blocked_map': blocked_map,
                # 'other_curr_nodes': [n.xy_name for n in other_curr_nodes],
                'curr_nodes': curr_n_name_to_agent_list,
            }
        return True, f'OK-1 - next_node {next_node.xy_name} is a goal node and it is free', 0, {}

    closest_corridor: List[Node] = build_corridor_from_nodes(curr_node, goal_node, nodes_dict, h_dict, non_sv_nodes_np)
    if closest_corridor[-1] == goal_node:
        if closest_corridor[-1].xy_name in curr_n_name_to_agent_list or blocked_map[closest_corridor[-1].x, closest_corridor[-1].y]:
            return False, f'PROBLEM-2 - last corridor node {goal_node.xy_name} is a goal node and is occupied or blocked', 2, {
                'goal_node': goal_node.xy_name,
                'closest_corridor': [n.xy_name for n in closest_corridor],
                'blocked_map': blocked_map,
                # 'other_curr_nodes': [n.xy_name for n in other_curr_nodes],
                'curr_nodes': curr_n_name_to_agent_list,
            }

    if full_corridor_check:
        if corridor_is_blocked_somewhere(closest_corridor, blocked_map):
            return False, f'PROBLEM-3 - part of the corridor is blocked: {[n.xy_name for n in closest_corridor if blocked_map[n.x, n.y]]}', 3, {
                'closest_corridor': [n.xy_name for n in closest_corridor],
                'blocked_map': blocked_map,
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
                    'blocked_map': blocked_map,
                    'open_list_names': open_list_names,
                    'closed_list_names': closed_list_names,
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
            nei_node = nodes_dict[nei_name]
            if blocked_map[nei_node.x, next_node.y]:
                touched_blocked_nodes = True
                touched_blocked_nodes_list.append(nei_name)
                continue
            open_list.append(nei_node)
            heapq.heappush(open_list_names, nei_name)
        heapq.heappush(closed_list_names, next_node.xy_name)

    error_num = 4 if touched_blocked_nodes else 5
    return False, f'PROBLEM-{error_num} - not_enough_free_nodes', error_num, {
        'closest_corridor': [n.xy_name for n in closest_corridor],
        'max_required_free_nodes': max_required_free_nodes,
        'free_count': free_count,
        'blocked_map': blocked_map,
        'open_list_names': open_list_names,
        'closed_list_names': closed_list_names,
        'touched_blocked_nodes': touched_blocked_nodes,
        'touched_blocked_nodes_list': touched_blocked_nodes_list
    }


def get_alter_goal_node(
        agent: AlgCgar3MapfAgent,
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        non_sv_nodes_with_blocked_np: np.ndarray,
        curr_n_name_to_agent_list: List[str],
        blocked_map: np.ndarray,
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
                blocked_map, full_corridor_check=True
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
            if blocked_map[n.x, n.y]:
                continue
            open_list.append(n)
        heapq.heappush(closed_list_names, alt_node.xy_name)
    return agent.goal_node


def corridor_is_blocked_somewhere(closest_corridor: List[Node], blocked_map: np.ndarray) -> bool:
    for n in closest_corridor:
        if blocked_map[n.x, n.y] == 1:
            return True
    return False


def build_vc_ec_from_configs(config_from: Dict[str, Node], config_to: Dict[str, Node]):
    vc_set, ec_set = [], []
    for agent_name, node_from in config_from.items():
        if agent_name in config_to:
            node_to = config_to[agent_name]
            heapq.heappush(vc_set, (node_to.x, node_to.y))
            heapq.heappush(ec_set, (node_from.x, node_from.y, node_to.x, node_to.y))
            # vc_set.append((node_to.x, node_to.y))
            # ec_set.append((node_from.x, node_from.y, node_to.x, node_to.y))
    # heapq.heapify(vc_set)
    # heapq.heapify(ec_set)
    return vc_set, ec_set


def procedure_i_pibt(
        agent: AlgCgar3MapfAgent,
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals: Dict[str, Node],
        curr_n_name_to_agent_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_agent_list: List[str],
        blocked_map: np.ndarray,
        boss_agent: AlgCgar3MapfAgent,
) -> bool:
    agent_name = agent.name
    agent_curr_node = config_from[agent_name]
    agent_goal_node = goals[agent_name]
    h_goal_np: np.ndarray = h_dict[agent_goal_node.xy_name]
    vc_set, ec_set = build_vc_ec_from_configs(config_from, config_to)

    # sort C in ascending order of dist(u, gi) where u ∈ C
    nei_nodes: List[Node] = [nodes_dict[n_name] for n_name in config_from[agent_name].neighbours]
    random.shuffle(nei_nodes)

    def get_nei_v(n: Node):
        # s_v = 0.5 if n.xy_name in curr_n_name_to_agent_list else 0
        s_v = 0
        return h_goal_np[n.x, n.y] + s_v

    nei_nodes.sort(key=get_nei_v)

    for j, nei_node in enumerate(nei_nodes):
        # vc
        if (nei_node.x, nei_node.y) in vc_set:
            continue
        # ec
        if (nei_node.x, nei_node.y, agent_curr_node.x, agent_curr_node.y) in ec_set:
            continue
        # blocked
        if blocked_map[nei_node.x, nei_node.y] and nei_node != agent_goal_node:
            continue

        # if nei_node in blocked_nodes and nei_node == agent_goal_node and agent.future_rank != 0:
        if blocked_map[nei_node.x, nei_node.y] and agent.curr_rank != boss_agent.curr_rank and nei_node == agent_goal_node:
            continue

        config_to[agent_name] = nei_node
        if nei_node.xy_name in curr_n_name_to_agent_list:
            next_agent = curr_n_name_to_agent_dict[nei_node.xy_name]
            if agent != next_agent and next_agent.name not in config_to:
                next_is_valid = procedure_i_pibt(
                    next_agent, nodes_dict, h_dict, config_from, config_to, goals,
                    curr_n_name_to_agent_dict, curr_n_name_to_agent_list, blocked_map, boss_agent
                )
                if not next_is_valid:
                    vc_set, ec_set = build_vc_ec_from_configs(config_from, config_to)
                    continue
        return True
    config_to[agent_name] = agent_curr_node
    return False


def run_i_pibt(
        main_agent: AlgCgar3MapfAgent,
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        config_from: Dict[str, Node] | None = None,
        config_to: Dict[str, Node] | None = None,
        goals: Dict[str, Node] | None = None,
        curr_n_name_to_agent_dict: Dict[str, AlgCgar3MapfAgent] | None = None,
        curr_n_name_to_agent_list: List[str] | None = None,
        blocked_map: np.ndarray | None = None,
) -> Dict[str, Node]:
    _ = procedure_i_pibt(main_agent, nodes_dict, h_dict, config_from, config_to, goals,
                         curr_n_name_to_agent_dict, curr_n_name_to_agent_list, blocked_map, boss_agent=main_agent)
    return config_to


def calc_pibt_step(
        main_agent: AlgCgar3MapfAgent, agents: List[AlgCgar3MapfAgent], nodes_dict: Dict[str, Node], h_dict: dict,
        given_goal_node: Node, blocked_map: np.ndarray, r_blocked_map: np.ndarray, config_from: Dict[str, Node], config_to: Dict[str, Node],
        goals: Dict[str, Node], curr_n_name_to_agent_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_agent_list: List[str], unplanned_agents: List[AlgCgar3MapfAgent], is_main_agent: bool = False,
        iteration: int = 0, to_assert: bool = False
) -> Dict[str, Node]:
    # print(f'\n --- inside calc_pibt_step {iteration} --- ')

    # Calc PIBT
    curr_blocked_map = np.copy(blocked_map)
    for agent in agents:
        i_curr = agent.curr_node
        if r_blocked_map[i_curr.x, i_curr.y] == 0:
            curr_blocked_map[i_curr.x, i_curr.y] = 0
    # if is_main_agent:
    curr_blocked_map[given_goal_node.x, given_goal_node.y] = 1
    config_to = run_i_pibt(
        main_agent=main_agent, nodes_dict=nodes_dict, h_dict=h_dict,
        config_from=config_from, config_to=config_to, goals=goals,
        curr_n_name_to_agent_dict=curr_n_name_to_agent_dict, curr_n_name_to_agent_list=curr_n_name_to_agent_list,
        blocked_map=curr_blocked_map)
    # Update paths
    for agent in unplanned_agents:
        if agent.name in config_to:
            next_node = config_to[agent.name]
            agent.path.append(next_node)

    return config_to


def calc_ep_steps(
        main_agent: AlgCgar3MapfAgent, agents: List[AlgCgar3MapfAgent], nodes: List[Node], nodes_dict: Dict[str, Node],
        h_dict: dict, given_goal_node: Node, config_from: Dict[str, Node],
        curr_n_name_to_agent_dict: Dict[str, AlgCgar3MapfAgent], curr_n_name_to_agent_list: List[str],
        a_non_sv_nodes_np: np.ndarray, blocked_map: np.ndarray, iteration: int
) -> None:
    """
    - Build corridor
    - Build EP for ev-agents in the corridor
    - Evacuate ev-agents
    - Build the steps in the corridor to the main agent
    """
    assert len(main_agent.path) == iteration

    # Preps
    # blocked_nodes_names: List[str] = [n.xy_name for n in blocked_nodes]
    # Calc
    corridor: List[Node] = build_corridor_from_nodes(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, a_non_sv_nodes_np
    )
    corridor_names: List[str] = [n.xy_name for n in corridor]
    assert corridor[0] == main_agent.path[-1]

    # if any of the corridor's nodes is blocked - just return
    assert len([n for n in corridor[1:] if blocked_map[n.x, n.y]]) == 0
        # return

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
    blocked_map[main_agent.curr_node.x, main_agent.curr_node.y] = 1
    blocked_map[given_goal_node.x, given_goal_node.y] = 1
    for ev_agent in ev_agents:
        ev_path, captured_free_node = find_ev_path_m(
            ev_agent.curr_node, corridor, nodes_dict, blocked_map, captured_free_nodes,
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
        parent_agent: AlgCgar3MapfAgent, agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]]
) -> None:
    if parent_agent is None:
        return
    prev_daddy_return_agents = agents_to_return_dict[parent_agent.name]
    assert parent_agent not in prev_daddy_return_agents
    for a in prev_daddy_return_agents:
        a.remove_return_road()
    init_a_in_atr_dict(agents_to_return_dict, parent_agent.name)


def update_agents_to_return(
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
                break
    if to_add:
        for p_agent in newly_planned_agents:
            # assert p_agent in lr_agents
            if p_agent not in agents_to_return:
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
        next_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
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
            # inside next_n_name_to_a_dict
            if n_name in next_n_name_to_a_dict:
                agent_on_road = next_n_name_to_a_dict[n_name]
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
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        future_captured_node_names: List[str],
        next_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        to_config: Dict[str, Node],
        iteration: int, to_assert: bool = False
) -> None:
    # ------------------------------------------------------------------ #
    def update_data(given_a: AlgCgar3MapfAgent, given_node: Node, to_pop: bool = False):
        to_config[given_a.name] = given_node
        next_n_name_to_a_dict[given_node.xy_name] = given_a
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
    update_waiting_tables(
        main_agent, hr_agents, lr_agents, planned_agents, agents_to_return, next_n_name_to_a_dict, future_captured_node_names,
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
        if next_possible_n_name in curr_n_name_to_a_dict:
            distur_agent = curr_n_name_to_a_dict[next_possible_node.xy_name]
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
        main_agent: AlgCgar3MapfAgent,
        agents_to_return: List[AlgCgar3MapfAgent], iteration: int
) -> Tuple[List[AlgCgar3MapfAgent], List[AlgCgar3MapfAgent]]:
    cleaned_agents_to_return: List[AlgCgar3MapfAgent] = []
    deleted_agents: List[AlgCgar3MapfAgent] = []
    for agent in agents_to_return:
        # assert len(agent.path) - 1 >= iteration
        if len(agent.return_road) == 1 and len(agent.path) - 1 == iteration:
            assert agent.return_road[-1][3] == agent.path[iteration]
            deleted_agents.append(agent)
        else:
            cleaned_agents_to_return.append(agent)
    for da in deleted_agents:
        da.remove_return_road()
    return cleaned_agents_to_return, deleted_agents


def remove_return_road_for_others(
    main_agent: AlgCgar3MapfAgent,
    agents_to_return: List[AlgCgar3MapfAgent],
    agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
    agents_dict: Dict[str, AlgCgar3MapfAgent]
):
    # if touching HR
    for atr in agents_to_return:
        atr_name = atr.name
        for k, v in agents_to_return_dict.items():
            if k == main_agent.name:
                continue
            if atr in v:
                k_agent = agents_dict[k]
                remove_return_paths_of_agent(k_agent, agents_to_return_dict)

    for a in agents_to_return:
        add_a_to_atr_dict(agents_to_return_dict, main_agent, a)


def if_anybody_crossed_me_remove_my_return_paths(
        main_agent: AlgCgar3MapfAgent,
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent], curr_n_name_to_a_list: List[str],
        next_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent], next_n_name_to_a_list: List[str],
        agents_to_return: List[AlgCgar3MapfAgent], iteration: int,
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
) -> None:
    for atr in agents_to_return:
        for n in atr.return_road_nodes:
            if n == main_agent.get_goal_node():
                remove_return_paths_of_agent(main_agent, agents_to_return_dict)
                return
            if n.xy_name in curr_n_name_to_a_list:
                a_on_road = curr_n_name_to_a_dict[n.xy_name]
                if a_on_road != main_agent:
                    if a_on_road not in agents_to_return:
                        # assert a_on_road.parent_of_path.curr_rank < main_agent.curr_rank
                        remove_return_paths_of_agent(main_agent, agents_to_return_dict)
                        return
            if n.xy_name in next_n_name_to_a_list:
                a_on_road = next_n_name_to_a_dict[n.xy_name]
                if a_on_road != main_agent:
                    if a_on_road not in agents_to_return:
                        # assert a_on_road.parent_of_path.curr_rank < main_agent.curr_rank
                        remove_return_paths_of_agent(main_agent, agents_to_return_dict)
                        return


def add_a_to_atr_dict(
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]], main_agent: AlgCgar3MapfAgent, agent_to_add: AlgCgar3MapfAgent
) -> None:
    prev_list = agents_to_return_dict[main_agent.name]
    if agent_to_add in prev_list:
        return
    i_main_agent_name = main_agent.name
    i_agent_to_add_name = agent_to_add.name
    for k, v in agents_to_return_dict.items():
        assert agent_to_add not in v
        # if agent_to_add in v:
        #     v.remove(agent_to_add)
    agents_to_return_dict[main_agent.name].append(agent_to_add)
    agent_to_add.set_parent_of_path(main_agent)


def init_a_in_atr_dict(
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]], key_name: str
) -> None:
    agents_to_return_dict[key_name] = []


def check_if_all_at_roads_are_in_list(
        main_agent: AlgCgar3MapfAgent,
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent], curr_n_name_to_a_list: List[str],
        next_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent], next_n_name_to_a_list: List[str],
        agents_to_return: List[AlgCgar3MapfAgent], iteration: int
):
    for atr in agents_to_return:
        for n in atr.return_road_nodes:
            if n.xy_name in curr_n_name_to_a_list:
                a_on_road = curr_n_name_to_a_dict[n.xy_name]
                if a_on_road != main_agent:
                    assert a_on_road in agents_to_return
            if n.xy_name in next_n_name_to_a_list:
                a_on_road = next_n_name_to_a_dict[n.xy_name]
                if a_on_road != main_agent:
                    assert a_on_road in agents_to_return
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
    # TODO: get rid of stay_where_you_are as much as possible
    # if the agent has a plan
    if main_agent.name in config_to:
        return True, {
            'message': 'in config_to', 'ccs_status': 1,
            'do_step_stage': False, 'do_return_stage': True
        }

    # If the agent is at its goal and has return paths to finish
    if main_agent.curr_node == main_agent.get_goal_node():
        main_a_return_agents_list = agents_to_return_dict[main_agent.name]
        if len(main_a_return_agents_list) > 0:
            stay_where_you_are(main_agent, config_to, iteration)
            return True, {
                'message': 'is at the goal, but need to wait for agents to return', 'ccs_status': 2,
                'do_step_stage': False, 'do_return_stage': True
            }
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
                stay_where_you_are(parent, config_to, iteration)
            # Change the goal of the agent i back to the original
            main_agent.remove_alt_goal_node()
        stay_where_you_are(main_agent, config_to, iteration)
        message = 'swap back' if the_order_swapped else ''
        return False, {
            'message': message, 'ccs_status': 3,
            'do_step_stage': False, 'do_return_stage': False
        }

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
            return True, {
                'message': 'pibt step', 'ccs_status': 4,
                'do_step_stage': True, 'do_return_stage': True
            }
    else:
        # EP step
        # If the next step/corridor is blocked somewhere
        if corridor_is_blocked_somewhere(closest_corridor, blocked_map):
            # stay_where_you_are(main_agent, config_to, iteration)
            return True, {
                'message': 'the next step/corridor is blocked somewhere', 'ccs_status': 5,
                'do_step_stage': False, 'do_return_stage': True
            }

    # blocked_nodes = [n for n in nodes if blocked_map[n.x, n.y] == 1]

    # (!) If the goal is in next step/corridor and is not blocked and is occupied by someone
    goal_in_corridor = given_goal_node in closest_corridor
    goal_not_blocked = blocked_map[given_goal_node.x, given_goal_node.y] == 0
    goal_is_occupied = given_goal_node.xy_name in curr_n_name_to_a_list
    if goal_in_corridor and goal_not_blocked and goal_is_occupied:
        distur_agent = curr_n_name_to_a_dict[given_goal_node.xy_name]
        alter_goal_node = get_alter_goal_node(
            distur_agent, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_a_list,
            blocked_map, goals=list(goals_dict.values()), avoid_curr_nodes=True, avoid_goals=True)
        distur_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        if distur_agent.alt_goal_node is not None:
            # assert distur_agent in lr_agents
            if distur_agent.future_rank > main_agent.future_rank:
                prev_main_priority = main_agent.future_rank
                main_agent.future_rank = distur_agent.future_rank
                distur_agent.future_rank = prev_main_priority
        stay_where_you_are(distur_agent, config_to, iteration)
        stay_where_you_are(main_agent, config_to, iteration)

        return True, {
            'message': 'swap', 'ccs_status': 6,
            'do_step_stage': False, 'do_return_stage': True
        }

    alt_is_good, alt_message, i_error, info = is_enough_free_locations(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_n_name_to_a_list, main_non_sv_nodes_np,
        blocked_map, full_corridor_check=True
    )
    # (!) If the goal is unreachable and the reason is not in blocked nodes
    if not alt_is_good and i_error == 5:
        alter_goal_node = get_alter_goal_node(
            main_agent, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_a_list,
            blocked_map, goals=list(goals_dict.values()), avoid_curr_nodes=True, avoid_goals=True)
        if alter_goal_node == main_agent.goal_node:
            stay_where_you_are(main_agent, config_to, iteration)
            return True, {
                'message': 'alt goal node is not good', 'ccs_status': 7,
                'do_step_stage': False, 'do_return_stage': True
            }
        stay_where_you_are(main_agent, config_to, iteration)
        main_agent.reset_alt_goal_node(alter_goal_node, main_agent)
        return True, {
            'message': 'error 5', 'ccs_status': 8,
            'do_step_stage': False, 'do_return_stage': True
        }

    # if any other reason to fail
    if not alt_is_good:
        stay_where_you_are(main_agent, config_to, iteration)
        return True, {
            'message': f'{alt_message}', 'ccs_status': 9,
            'do_step_stage': False, 'do_return_stage': True
        }

    return True, {
        'message': 'continue', 'ccs_status': 10,
        'do_step_stage': True, 'do_return_stage': True
    }


def calc_step_stage(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        blocked_map: np.ndarray,
        r_blocked_map: np.ndarray,
        iteration: int,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals_dict: Dict[str, Node],
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_a_list: List[str],
        check_stage_info: dict,
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

    if not check_stage_info['do_step_stage']:
        message = check_stage_info['message']
        main_agent.last_calc_step_message = message
        return message
    # ---------------------------------------------------------------------------------------------------------- #
    # EXECUTE THE FORWARD STEP
    # ---------------------------------------------------------------------------------------------------------- #
    unplanned_agents = [a for a in agents if a.name not in config_to]
    # decide on the goal
    given_goal_node = main_agent.get_goal_node()
    a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
    # blocked_map: np.ndarray = get_blocked_map(main_agent, hr_agents, lr_agents, agents, agents_to_return_dict, img_np, iteration)
    # blocked_nodes = get_blocked_nodes_from_map(nodes, nodes_dict, blocked_map)

    a_next_node = get_min_h_nei_node(main_agent.curr_node, given_goal_node, nodes_dict, h_dict)
    if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
        # calc single PIBT step
        # blocked_nodes = get_blocked_nodes(self.agents, iteration, self.need_to_freeze_main_goal_node)
        calc_pibt_step(main_agent, agents, nodes_dict, h_dict, given_goal_node, blocked_map, r_blocked_map, config_from, config_to,
                       goals_dict, curr_n_name_to_a_dict, curr_n_name_to_a_list, unplanned_agents,
                       iteration=iteration)
        message = f'plan of pibt in i {iteration}'
    else:
        # calc evacuation of agents from the corridor
        calc_ep_steps(main_agent, agents, nodes, nodes_dict, h_dict, given_goal_node, config_from,
                      curr_n_name_to_a_dict, curr_n_name_to_a_list, a_non_sv_nodes_np,
                      blocked_map, iteration)
        message = f'plan of ev in i {iteration}'

    update_config_to(config_to, unplanned_agents, iteration)

    main_agent.last_calc_step_message = message
    newly_planned_agents = [a for a in unplanned_agents if a.name in config_to]
    # if len(newly_planned_agents) > 0:
    #     assert main_agent in newly_planned_agents
    update_parent_of_path(newly_planned_agents, parent=main_agent)
    return message


def return_agents_stage(
        main_agent: AlgCgar3MapfAgent,
        hr_agents: List[AlgCgar3MapfAgent],
        lr_agents: List[AlgCgar3MapfAgent],
        iteration: int,
        check_stage_info: dict,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals_dict: Dict[str, Node],
        curr_n_name_to_a_dict: Dict[str, AlgCgar3MapfAgent],
        curr_n_name_to_a_list: List[str],
        newly_planned_agents: List[AlgCgar3MapfAgent],
        future_captured_node_names: List[str],
        agents: List[AlgCgar3MapfAgent],
        agents_dict: Dict[str, AlgCgar3MapfAgent],
        agents_to_return_dict: Dict[str, List[AlgCgar3MapfAgent]],
) -> str:
    if not check_stage_info['do_return_stage']:
        message = check_stage_info['message']
        return message
    # if the agent is not the boss of its life -> return
    if main_agent.parent_of_path != main_agent:
        remove_return_paths_of_agent(main_agent, agents_to_return_dict)
        return 'main_agent is no longer a parent of himself'

    agents_to_return = agents_to_return_dict[main_agent.name]
    agents_to_return = update_agents_to_return(agents_to_return, newly_planned_agents, iteration)

    remove_return_road_for_others(main_agent, agents_to_return, agents_to_return_dict, agents_dict)

    agents_to_return = agents_to_return_dict[main_agent.name]

    if len(agents_to_return) == 0:
        return '(1) len(agents_to_return) == 0'
    assert main_agent not in agents_to_return

    planned_agents = [a for a in agents_to_return if len(a.path) - 1 >= iteration]
    backward_step_agents = [a for a in agents_to_return if len(a.path) - 1 == iteration - 1]
    next_n_name_to_a_dict = {node.xy_name: agents_dict[agent_name] for agent_name, node in config_to.items()}
    next_n_name_to_a_list = list(next_n_name_to_a_dict.keys())
    heapq.heapify(next_n_name_to_a_list)

    all_update_return_roads(planned_agents, iteration)

    if_anybody_crossed_me_remove_my_return_paths(
        main_agent,
        curr_n_name_to_a_dict, curr_n_name_to_a_list,
        next_n_name_to_a_dict, next_n_name_to_a_list,
        agents_to_return, iteration, agents_to_return_dict
    )
    agents_to_return = agents_to_return_dict[main_agent.name]


    if len(agents_to_return) == 0:
        return '(2) len(agents_to_return) == 0'

    # check_if_all_at_roads_are_in_list(
    #     main_agent,
    #     curr_n_name_to_a_dict, curr_n_name_to_a_list,
    #     next_n_name_to_a_dict, next_n_name_to_a_list,
    #     agents_to_return, iteration
    # )

    calc_backward_road(
        main_agent, hr_agents, lr_agents, backward_step_agents, planned_agents, agents_to_return, agents_dict, curr_n_name_to_a_dict,
        future_captured_node_names, next_n_name_to_a_dict, config_to, iteration,
    )

    agents_to_return, deleted_agents = clean_agents_to_return(main_agent, agents_to_return, iteration)
    init_a_in_atr_dict(agents_to_return_dict, main_agent.name)
    for a in agents_to_return:
        add_a_to_atr_dict(agents_to_return_dict, main_agent, a)

    # update_config_to(config_to, agents, iteration)
    update_config_to(config_to, backward_step_agents, iteration)

    return 'returned someone'





















