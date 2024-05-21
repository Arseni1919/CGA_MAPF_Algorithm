import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt
from algs.alg_CGAR import align_all_paths, get_min_h_nei_node
from algs.alg_CGAR import build_corridor, find_ev_path, push_ev_agents, push_main_agent, update_waiting_tables
from algs.alg_CGAR_Seq_MAPF import is_enough_free_locations
from algs.alg_CGA_MAPF import get_alter_goal_node


class AlgCgarMapfAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node, nodes: List[Node], nodes_dict: Dict[str, Node]):
        self.num = num
        self.priority = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.alt_goal_node: Node | None = None
        self.setting_agent_name: str | None = None
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.path: List[Node] = [start_node]
        self.first_arrived: bool = self.curr_node == self.goal_node
        if self.first_arrived:
            # Tuple: (0) node name, (1) iteration, (2) list of names of agents that will wait for you, (3) the node
            self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque([(self.goal_node.xy_name, 0, [], self.goal_node)])
        else:
            self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque()
        self.trash_return_road: Deque[Tuple[str, int, List[Self], Node]] = deque()
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

    def execute_forward_step(self, iteration: int) -> None:
        # execute the step
        self.prev_node = self.curr_node
        self.curr_node = self.path[iteration]
        assert self.prev_node.xy_name in self.curr_node.neighbours
        if self.alt_goal_node is not None and self.curr_node == self.alt_goal_node:
            self.reset_alt_goal_node()

        if not self.first_arrived and self.curr_node == self.goal_node and self.priority == 0:
            self.first_arrived = True
            self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque([(self.goal_node.xy_name, iteration, [], self.goal_node)])
            self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}

        # for the back-steps
        if self.first_arrived:
            if self.curr_node != self.return_road[-1][3]:
                self.return_road.append((self.curr_node.xy_name, iteration, [], self.curr_node))

    def execute_simple_step(self, iteration: int) -> None:
        # execute the step
        self.prev_node = self.curr_node
        self.curr_node = self.path[iteration]
        assert self.prev_node.xy_name in self.curr_node.neighbours
        if self.alt_goal_node is not None and self.curr_node == self.alt_goal_node:
            self.reset_alt_goal_node()

    def get_goal_node(self) -> Node:
        if self.alt_goal_node is None:
            return self.goal_node
        return self.alt_goal_node

    def reset_alt_goal_node(self, node: Node | None = None, setting_agent: Self | None = None) -> None:
        if node is not None:
            assert setting_agent.priority <= self.priority
            self.alt_goal_node = node
            self.setting_agent_name = setting_agent.name
        else:
            self.alt_goal_node = None
            self.setting_agent_name = None

    def reset_return_road(self, iteration):
        if self.first_arrived:
            if self.curr_node == self.goal_node:
                self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque([(self.goal_node.xy_name, iteration, [], self.goal_node)])
            else:
                self.first_arrived = False
                self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque()
            self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}

    def get_wl(self, node: Node, to_assert: bool = False):
        return self.waiting_table[node.xy_name]

    def get_wl_names(self, node: Node, to_assert: bool = False):
        return [tpl[0] for tpl in self.waiting_table[node.xy_name]]

    def add_to_wl(self, node: Node, agent_on_road: Self, iteration: int, to_assert: bool = False):
        # if to_assert:
        if (agent_on_road.name, iteration) not in self.waiting_table[node.xy_name]:
            self.waiting_table[node.xy_name].append((agent_on_road.name, iteration))

    def remove_from_wl(self, node: Node, agent_on_road: Self, iteration: int, to_assert: bool = False):
        remove_item = (agent_on_road.name, iteration)
        # assert remove_item in self.waiting_table[node.xy_name]
        self.waiting_table[node.xy_name] = [i for i in self.waiting_table[node.xy_name] if i != remove_item]

    def add_affected_agent_to_rr(self, affected_agent: Self, iteration: int, n_name: str, n: Node,
                                 to_assert: bool = False):
        my_n_name, my_i, my_a_list, my_n = self.return_road[-1]
        my_a_list.append(affected_agent.name)
        # if to_assert:
        assert my_i == iteration
        assert my_n_name == n_name
        assert my_n == n



def get_blocked_nodes_for_pibt(agents: List[AlgCgarMapfAgent], iteration: int, need_to_freeze_main_goal_node: bool = False, backward_step_agents: List[AlgCgarMapfAgent] | None = None) -> List[Node]:
    blocked_nodes: List[Node] = []

    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            for n in agent.path[iteration:]:
                blocked_nodes.append(n)

    if backward_step_agents is not None:
        for agent in backward_step_agents:
            for n in agent.return_road_nodes:
                blocked_nodes.append(n)

    if need_to_freeze_main_goal_node:
        main_agent = agents[0]
        blocked_nodes.append(main_agent.get_goal_node())

    blocked_nodes = list(set(blocked_nodes))
    return blocked_nodes


def get_blocked_nodes_for_ev(agents: List[AlgCgarMapfAgent], iteration: int, need_to_freeze_main_goal_node: bool = False, backward_step_agents: List[AlgCgarMapfAgent] | None = None) -> List[Node]:
    blocked_nodes: List[Node] = []

    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            captured_nodes = agent.path[iteration - 1:]
            blocked_nodes.extend(captured_nodes)

    if backward_step_agents is not None:
        for agent in backward_step_agents:
            for n in agent.return_road_nodes:
                blocked_nodes.append(n)

    if need_to_freeze_main_goal_node:
        main_agent = agents[0]
        blocked_nodes.append(main_agent.get_goal_node())

    blocked_nodes = list(set(blocked_nodes))
    return blocked_nodes


def all_reset_return_roads(agents: List[AlgCgarMapfAgent], iteration: int):
    for agent in agents:
        agent.reset_return_road(iteration)


def get_future_captured_node_names(agents: List[AlgCgarMapfAgent], iteration: int) -> Tuple[List[str], List[AlgCgarMapfAgent], List[AlgCgarMapfAgent], Dict[str, AlgCgarMapfAgent], Dict[str, Node]]:
    future_captured_node_names: List[str] = []
    forward_step_agents: List[AlgCgarMapfAgent] = []
    backward_step_agents: List[AlgCgarMapfAgent] = []
    fs_to_a_dict: Dict[str, AlgCgarMapfAgent] = {}
    to_config: Dict[str, Node] = {}
    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            forward_step_agents.append(agent)
            fs_to_a_dict[agent.path[iteration].xy_name] = agent
            to_config[agent.name] = agent.path[iteration]
            for n in agent.path[iteration:]:
                heapq.heappush(future_captured_node_names, n.xy_name)
        if len(agent.path) - 1 == iteration - 1 and agent.first_arrived:
            backward_step_agents.append(agent)
        assert not len(agent.path) - 1 < iteration - 1
        return future_captured_node_names, forward_step_agents, backward_step_agents, fs_to_a_dict, to_config


def calc_backward_road[T](
        from_n_to_a_dict: Dict[str, T], backward_step_agents: List[T], future_captured_node_names: List[str],
        fs_to_a_dict: Dict[str, T], to_config: Dict[str, Node],  agents: List[T], agents_dict: Dict[str, T],
        iteration: int, to_assert: bool = False
) -> None:
    # ------------------------------------------------------------------ #
    def update_data(given_a: T, given_node: Node, to_pop: bool = False):
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

    update_waiting_tables(agents, fs_to_a_dict, from_n_to_a_dict, iteration, to_assert)

    # decide rest of to_config
    open_list: Deque[T] = deque(agents[:])
    while len(open_list) > 0:
        next_agent = open_list.popleft()
        # already planned
        if next_agent.name in to_config:
            continue
        # did not arrived to the goal, therefore no need to return - calc later
        if not next_agent.first_arrived:
            continue
        # no need to return, the agent wasn't displaced
        if len(next_agent.return_road) == 1:
            continue
        next_possible_n_name, next_rr_i, next_rr_a_list, next_possible_node = next_agent.return_road[-2]
        # next possible move is not allowed
        if next_possible_node.xy_name in future_captured_node_names:
            update_data(next_agent, next_agent.curr_node)
            continue
        # another agent in front of the agent needs to plan first
        # Circles - there will be never circles here = with the circles the algorithm will not work
        if next_possible_node.xy_name in from_n_to_a_dict:
            distur_agent = from_n_to_a_dict[next_possible_node.xy_name]
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
        assert len(agent.path) == iteration
        to_node = to_config[agent.name]
        agent.path.append(to_node)
    return


def get_goal_location_is_occupied(
        main_agent: AlgCgarMapfAgent, node_name_to_agent_dict: Dict[str, AlgCgarMapfAgent], node_name_to_agent_list: List[str]
) -> Tuple[bool, AlgCgarMapfAgent | None]:
    assert main_agent.priority == 0
    given_goal_node_name = main_agent.get_goal_node().xy_name
    if main_agent.get_goal_node().xy_name in node_name_to_agent_list:
        distur_a = node_name_to_agent_dict[main_agent.get_goal_node().xy_name]
        return True, distur_a
    return False, None


def liberate_goal_location(
        main_agent: AlgCgarMapfAgent,
        distur_a: AlgCgarMapfAgent,
        agents: List[AlgCgarMapfAgent],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        curr_nodes: List[Node],
        non_sv_nodes_with_blocked_np: np.ndarray,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        node_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
        node_name_to_agent_list: List[str],
        need_to_freeze_main_goal_node: bool,
        iteration: int,
        to_assert: bool):
    if len(distur_a.path) - 1 >= iteration:
        return [], []
    if distur_a.alt_goal_node is not None:
        assert distur_a.setting_agent_name == main_agent.name
    blocked_nodes = get_blocked_nodes_for_ev(agents, iteration)
    goals_list: List[Node] = [a.get_goal_node() for a in agents if a != distur_a]
    blocked_nodes.extend(goals_list)
    distur_a_alter_goal_node = get_alter_goal_node(distur_a, nodes_dict, h_dict, curr_nodes,
                                                   non_sv_nodes_with_blocked_np, blocked_nodes)
    distur_a.reset_alt_goal_node(distur_a_alter_goal_node, main_agent)

    goals = {a.name: a.get_goal_node() for a in agents}
    regular_agent_decision(
        distur_a,
        agents,
        nodes,
        nodes_dict,
        h_dict,
        config_from,
        config_to,
        goals,
        node_name_to_agent_dict,
        node_name_to_agent_list,
        [],
        non_sv_nodes_with_blocked_np,
        need_to_freeze_main_goal_node,
        curr_nodes,
        iteration, to_assert)


def regular_agent_decision(
        agent: AlgCgarMapfAgent,
        agents: List[AlgCgarMapfAgent],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals: Dict[str, Node],
        node_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
        node_name_to_agent_list: List[str],
        backward_step_agents: List[AlgCgarMapfAgent],
        non_sv_nodes_with_blocked_np: np.ndarray,
        need_to_freeze_main_goal_node: bool,
        curr_nodes: List[Node],
        iteration: int, to_assert: bool
    ) -> None:
        # decide on the goal
        given_goal_node = agent.get_goal_node()
        a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
        blocked_nodes = get_blocked_nodes_for_pibt(agents, iteration, need_to_freeze_main_goal_node, backward_step_agents)
        is_good, message, i_error, info = is_enough_free_locations(
            agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_nodes, a_non_sv_nodes_np,
            blocked_nodes)
        if not is_good:
            return

        a_next_node = get_min_h_nei_node(agent.curr_node, given_goal_node, nodes_dict, h_dict)
        if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
            # calc single PIBT step
            # blocked_nodes = get_blocked_nodes(self.agents, iteration, self.need_to_freeze_main_goal_node)
            calc_pibt_step(agent, agents, nodes_dict, h_dict, given_goal_node, blocked_nodes, config_from, config_to,
                           goals, node_name_to_agent_dict, node_name_to_agent_list, iteration, to_assert=to_assert)
        else:
            # calc evacuation of agents from the corridor
            calc_ep_steps(agent, agents, nodes, nodes_dict, h_dict, given_goal_node, config_from,
                               node_name_to_agent_dict, node_name_to_agent_list, a_non_sv_nodes_np, backward_step_agents, need_to_freeze_main_goal_node,
                               iteration, to_assert=to_assert)


def calc_pibt_step(main_agent: AlgCgarMapfAgent, agents: List[AlgCgarMapfAgent], nodes_dict: Dict[str, Node], h_dict: dict, given_goal_node: Node, blocked_nodes: List[Node],
                   config_from: Dict[str, Node], config_to: Dict[str, Node], goals: Dict[str, Node],
                   node_name_to_agent_dict: Dict[str, AlgCgarMapfAgent], node_name_to_agent_list: List[str],
                   iteration: int, to_assert: bool = False) -> Dict[str, Node]:
    # print(f'\n --- inside calc_pibt_step {iteration} --- ')
    assert len(main_agent.path) == iteration

    # Preps
    # config_to = {}
    for agent in agents:
        if agent.name not in config_to and len(agent.path) - 1 >= iteration:
            config_to[agent.name] = agent.path[iteration]
    if to_assert:
        assert len(set(config_to.values())) == len(set(config_to.keys()))

    # Calc PIBT
    curr_blocked_nodes = blocked_nodes[:]
    curr_blocked_nodes.append(given_goal_node)
    config_to = run_i_pibt(
        main_agent=main_agent, agents=agents, nodes_dict=nodes_dict, h_dict=h_dict,
        config_from=config_from, config_to=config_to, goals=goals,
        node_name_to_agent_dict=node_name_to_agent_dict, node_name_to_agent_list=node_name_to_agent_list,
        blocked_nodes=curr_blocked_nodes, given_goal_node=given_goal_node)

    # Update paths
    for agent in agents:
        if len(agent.path) == iteration and agent.name in config_to:
            next_node = config_to[agent.name]
            agent.path.append(next_node)

    return config_to


def calc_ep_steps(main_agent: AlgCgarMapfAgent, agents: List[AlgCgarMapfAgent], nodes: List[Node], nodes_dict: Dict[str, Node], h_dict: dict, given_goal_node: Node, config_from: Dict[str, Node],
                  node_name_to_agent_dict: Dict[str, AlgCgarMapfAgent], node_name_to_agent_list: List[str],
                  a_non_sv_nodes_np: np.ndarray, backward_step_agents: List[AlgCgarMapfAgent], need_to_freeze_main_goal_node: bool,
                  iteration: int, to_assert: bool = False) -> None:
    """
    - Build corridor
    - Build EP for ev-agents in the corridor
    - Evacuate ev-agents
    - Build the steps in the corridor to the main agent
    """
    assert len(main_agent.path) == iteration

    # Preps
    if main_agent.priority == 0:
        blocked_nodes = get_blocked_nodes_for_ev(agents, iteration)
        # assert main_agent.get_goal_node() not in blocked_nodes
    else:
        blocked_nodes = get_blocked_nodes_for_ev(agents, iteration, need_to_freeze_main_goal_node, backward_step_agents)
    blocked_nodes_names: List[str] = [n.xy_name for n in blocked_nodes]
    # Calc
    corridor: List[Node] = build_corridor(main_agent, nodes_dict, h_dict, a_non_sv_nodes_np,
                                          given_goal_node=given_goal_node)
    corridor_names: List[str] = [n.xy_name for n in corridor]
    assert corridor[0] == main_agent.path[-1]

    # if any of the corridor's nodes is blocked - just return
    if len([n for n in corridor[1:] if n in blocked_nodes]) > 0:
        return

    assert not (corridor[-1] == given_goal_node and corridor[-1].xy_name in node_name_to_agent_dict)

    # Find ev-agents
    ev_agents: List[AlgCgarMapfAgent] = []
    for node in corridor[1:]:
        if node.xy_name in node_name_to_agent_list:
            ev_agent = node_name_to_agent_dict[node.xy_name]
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
            node_name_to_agent_dict, node_name_to_agent_list
        )
        if ev_path is None:
            return
        captured_free_nodes.append(captured_free_node)
        ev_paths_list.append(ev_path)

    # Build steps for the ev-agents inside the ev-paths + extend paths
    moved_agents = []
    last_visit_dict = {n.xy_name: 0 for n in nodes}
    for i_ev_path, ev_path in enumerate(ev_paths_list):
        curr_n_name_to_a_dict: Dict[str, AlgCgarMapfAgent] = {a.path[-1].xy_name: a for a in agents}
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


def main_agent_decision(
        main_agent: AlgCgarMapfAgent,
        agents: List[AlgCgarMapfAgent],
        agents_dict: Dict[str, AlgCgarMapfAgent],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        curr_nodes: List[Node],
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals: Dict[str, Node],
        node_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
        node_name_to_agent_list: List[str],
        non_sv_nodes_with_blocked_np: np.ndarray,
        iteration: int, to_assert: bool
) -> Tuple[List[AlgCgarMapfAgent], List[AlgCgarMapfAgent]]:

    assert main_agent.priority == 0
    given_goal_node = main_agent.get_goal_node()
    a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
    blocked_nodes = get_blocked_nodes_for_ev(agents, iteration)
    # ---------------------------------------------------------------------------------------------------------- #
    # deal with failures
    # ---------------------------------------------------------------------------------------------------------- #
    is_good, message, i_error, info = is_enough_free_locations(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_nodes, a_non_sv_nodes_np,
        blocked_nodes, full_corridor_check=True)
    if not is_good:
        # THERE IS AN AGENT IN THE GOAL LOCATION
        assert i_error not in [1, 2]
        # THE SEARCH OF THE CORRIDOR IS BLOCKED BY PATHS OF OTHER AGENTS
        if i_error in [3, 4]:
            print(f'\n{i_error=}, {message}')
            main_agent.path.append(main_agent.path[-1])
            return [], []
        # THE GOAL IS UNREACHABLE - NEED TO CHANGE START LOCATION
        if i_error in [5]:
            print(f'\n{i_error=}, {message}')
            a_alter_goal_node = get_alter_goal_node(
                main_agent, nodes_dict, h_dict, curr_nodes, non_sv_nodes_with_blocked_np,
                blocked_nodes, full_corridor_check=True)
            main_agent.reset_alt_goal_node(a_alter_goal_node, main_agent)
            given_goal_node = main_agent.get_goal_node()
            a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]

    # ---------------------------------------------------------------------------------------------------------- #
    # EXECUTE THE FORWARD STEP
    # ---------------------------------------------------------------------------------------------------------- #
    a_next_node = get_min_h_nei_node(main_agent.curr_node, given_goal_node, nodes_dict, h_dict)
    if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
        # calc single PIBT step
        blocked_nodes = get_blocked_nodes_for_pibt(agents, iteration)
        calc_pibt_step(main_agent, agents, nodes_dict, h_dict, given_goal_node, blocked_nodes, config_from, config_to,
                       goals, node_name_to_agent_dict, node_name_to_agent_list, iteration, to_assert=to_assert)
    else:
        # calc evacuation of agents from the corridor
        calc_ep_steps(main_agent, agents, nodes, nodes_dict, h_dict, given_goal_node, config_from,
                      node_name_to_agent_dict, node_name_to_agent_list, a_non_sv_nodes_np, [],
                      need_to_freeze_main_goal_node=False,
                      iteration=iteration, to_assert=to_assert)
    # ---------------------------------------------------------------------------------------------------------- #
    # EXECUTE THE BACKWARD STEP
    # ---------------------------------------------------------------------------------------------------------- #
    # future_captured_node_names, forward_step_agents, backward_step_agents, fs_to_a_dict, to_config = get_future_captured_node_names(agents, iteration)
    # calc_backward_road(
    #     node_name_to_agent_dict, backward_step_agents, future_captured_node_names, fs_to_a_dict, to_config,
    #     agents, agents_dict, iteration, to_assert=to_assert
    # )
    # return forward_step_agents, backward_step_agents
    return [], []





