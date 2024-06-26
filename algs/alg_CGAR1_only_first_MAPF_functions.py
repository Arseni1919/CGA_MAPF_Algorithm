import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_b_PIBT import run_i_pibt
from algs.alg_CGAR import align_all_paths, get_min_h_nei_node
from algs.alg_CGAR import build_corridor, find_ev_path, push_ev_agents, push_main_agent, build_corridor_from_nodes


class AlgCgarMapfAgent:
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
        assert setting_agent.priority <= self.priority
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


def get_blocked_nodes_for_pibt(agents: List[AlgCgarMapfAgent], iteration: int, need_to_freeze_main_goal_node: bool = False, backward_step_agents: List[AlgCgarMapfAgent] | None = None) -> List[Node]:
    blocked_nodes: List[Node] = []

    for agent in agents:
        if len(agent.path) - 1 >= iteration:
            for n in agent.path[iteration:]:
                blocked_nodes.append(n)

    if backward_step_agents is not None:
        for agent in backward_step_agents:
            blocked_nodes.extend(agent.return_road_nodes)

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
            blocked_nodes.extend(agent.return_road_nodes)

    if need_to_freeze_main_goal_node:
        main_agent = agents[0]
        blocked_nodes.append(main_agent.get_goal_node())

    blocked_nodes = list(set(blocked_nodes))
    return blocked_nodes


def get_goal_location_is_occupied(
        main_agent: AlgCgarMapfAgent, curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent], curr_n_name_to_agent_list: List[str]
) -> Tuple[bool, AlgCgarMapfAgent | None]:
    assert main_agent.priority == 0
    given_goal_node_name = main_agent.get_goal_node().xy_name
    if main_agent.get_goal_node().xy_name in curr_n_name_to_agent_list:
        distur_a = curr_n_name_to_agent_dict[main_agent.get_goal_node().xy_name]
        if distur_a != main_agent:
            return True, distur_a
    return False, None


def regular_agent_decision(
        agent: AlgCgarMapfAgent,
        agents: List[AlgCgarMapfAgent],
        agents_dict: Dict[str, AlgCgarMapfAgent],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals: Dict[str, Node],
        curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
        curr_n_name_to_agent_list: List[str],
        backward_step_agents: List[AlgCgarMapfAgent],
        non_sv_nodes_with_blocked_np: np.ndarray,
        need_to_freeze_main_goal_node: bool,
        iteration: int, to_assert: bool
) -> str:
        # decide on the goal
        given_goal_node = agent.get_goal_node()
        a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
        blocked_nodes = get_blocked_nodes_for_pibt(
            agents, iteration, need_to_freeze_main_goal_node, backward_step_agents
        )
        is_good, message, i_error, info = is_enough_free_locations(
            agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_n_name_to_agent_list, a_non_sv_nodes_np,
            blocked_nodes)
        if not is_good:
            return f'no plan | {message}'

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


def one_of_best_next_nodes_are_blocked(
        main_agent: AlgCgarMapfAgent, given_goal_node: Node, blocked_nodes: List[Node],
        nodes_dict: Dict[str, Node], h_dict: dict
) -> bool:
    # a_next_node = get_min_h_nei_node(main_agent.curr_node, given_goal_node, nodes_dict, h_dict)
    nei_nodes = [nodes_dict[nn] for nn in main_agent.curr_node.neighbours]
    goal_h_np: np.ndarray = h_dict[given_goal_node.xy_name]
    min_h = min([goal_h_np[n.x, n.y] for n in nei_nodes])
    a_next_nodes = [n for n in nei_nodes if goal_h_np[n.x, n.y] == min_h]
    for a_next_node in a_next_nodes:
        if a_next_node in blocked_nodes:
            return True
    return False


def calc_pibt_step(
        main_agent: AlgCgarMapfAgent, agents: List[AlgCgarMapfAgent], nodes_dict: Dict[str, Node], h_dict: dict,
        given_goal_node: Node, blocked_nodes: List[Node], config_from: Dict[str, Node], config_to: Dict[str, Node],
        goals: Dict[str, Node], curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
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
        main_agent: AlgCgarMapfAgent, agents: List[AlgCgarMapfAgent], nodes: List[Node], nodes_dict: Dict[str, Node],
        h_dict: dict, given_goal_node: Node, config_from: Dict[str, Node],
        curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent], curr_n_name_to_agent_list: List[str],
        a_non_sv_nodes_np: np.ndarray, backward_step_agents: List[AlgCgarMapfAgent],
        need_to_freeze_main_goal_node: bool, iteration: int, to_assert: bool = False
) -> None:
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

    assert not (corridor[-1] == given_goal_node and corridor[-1].xy_name in curr_n_name_to_agent_dict)

    # Find ev-agents
    ev_agents: List[AlgCgarMapfAgent] = []
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


def main_stay_on_goal_check_and_update(main_agent: AlgCgarMapfAgent, iteration: int) -> None:
    if main_agent.curr_node == main_agent.get_goal_node() and len(main_agent.path) - 1 == iteration - 1:
        assert main_agent.path[-1] == main_agent.curr_node
        main_agent.path.append(main_agent.curr_node)


def main_agent_decision(
        main_agent: AlgCgarMapfAgent,
        agents: List[AlgCgarMapfAgent],
        agents_dict: Dict[str, AlgCgarMapfAgent],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals: Dict[str, Node],
        curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
        curr_n_name_to_agent_list: List[str],
        non_sv_nodes_with_blocked_np: np.ndarray,
        iteration: int, to_assert: bool
) -> str:

    assert main_agent.priority == 0
    given_goal_node = main_agent.get_goal_node()
    a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
    blocked_nodes = get_blocked_nodes_for_ev(agents, iteration)
    # ---------------------------------------------------------------------------------------------------------- #
    # if has a plan -> return | if at its goal -> just stay at place
    # ---------------------------------------------------------------------------------------------------------- #
    main_stay_on_goal_check_and_update(main_agent, iteration)
    has_a_plan = len(main_agent.path) - 1 >= iteration  # already planned
    on_its_goal = main_agent.curr_node == main_agent.get_goal_node()  # already at its goal
    if has_a_plan or on_its_goal:
        return f'{has_a_plan=} | {on_its_goal=}'
    # ---------------------------------------------------------------------------------------------------------- #
    # deal with failures
    # ---------------------------------------------------------------------------------------------------------- #
    is_good, message, i_error, info = is_enough_free_locations(
        main_agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_n_name_to_agent_list, a_non_sv_nodes_np,
        blocked_nodes, full_corridor_check=True)
    if not is_good:
        # THERE IS AN AGENT IN THE GOAL LOCATION
        assert i_error not in [1, 2], message
        # THE SEARCH OF THE CORRIDOR IS BLOCKED BY PATHS OF OTHER AGENTS
        if i_error in [3, 4]:
            print(f'\n{i_error=}, {message}')
            main_agent.path.append(main_agent.path[-1])
            return f'{message}'
        # THE GOAL IS UNREACHABLE - NEED TO CHANGE START LOCATION
        # assert i_error not in [5], message
        if i_error in [5]:
            print(f'\n{i_error=}, {message}', end='')
            a_alter_goal_node = get_alter_goal_node(
                main_agent, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_agent_list,
                blocked_nodes, full_corridor_check=True, avoid_curr_nodes=True)
            main_agent.reset_alt_goal_node(a_alter_goal_node, main_agent)
            given_goal_node = main_agent.get_goal_node()
            a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
            print(f' ...alternative goal is: {a_alter_goal_node.xy_name}')

    # ---------------------------------------------------------------------------------------------------------- #
    # EXECUTE THE FORWARD STEP
    # ---------------------------------------------------------------------------------------------------------- #
    a_next_node = get_min_h_nei_node(main_agent.curr_node, given_goal_node, nodes_dict, h_dict)
    if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
        # calc single PIBT step
        blocked_nodes = get_blocked_nodes_for_pibt(agents, iteration)
        calc_pibt_step(main_agent, agents, nodes_dict, h_dict, given_goal_node, blocked_nodes, config_from, config_to,
                       goals, curr_n_name_to_agent_dict, curr_n_name_to_agent_list, is_main_agent=True,
                       iteration=iteration, to_assert=to_assert)
    else:
        # calc evacuation of agents from the corridor
        calc_ep_steps(main_agent, agents, nodes, nodes_dict, h_dict, given_goal_node, config_from,
                      curr_n_name_to_agent_dict, curr_n_name_to_agent_list, a_non_sv_nodes_np, [],
                      need_to_freeze_main_goal_node=False,
                      iteration=iteration, to_assert=to_assert)
    return 'main agent finished to plan'


def update_priority_numbers(agents: List[AlgCgarMapfAgent]):
    for i_priority, agent in enumerate(agents):
        agent.priority = i_priority


def reset_the_first_agent_if_goal_occupied(
        agents: List[AlgCgarMapfAgent],
        nodes_dict: Dict[str, Node],
        h_dict: dict,
        curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
        curr_n_name_to_agent_list: List[str],
        non_sv_nodes_with_blocked_np: np.ndarray,
        iteration: int
) -> List[AlgCgarMapfAgent]:
    # ---------------------------------------------------------------------------------------------------- #
    # Liberate the goal location and freeze
    # ---------------------------------------------------------------------------------------------------- #
    main_agent = agents[0]
    # if main_agent
    goal_location_is_occupied, distur_a = get_goal_location_is_occupied(main_agent, curr_n_name_to_agent_dict,
                                                                        curr_n_name_to_agent_list)
    if goal_location_is_occupied:
        # blocked_nodes = get_blocked_nodes_for_ev(agents, iteration)
        blocked_nodes = []
        goals_list = [a.get_goal_node() for a in agents if a != distur_a]
        distur_a_alter_goal_node = get_alter_goal_node(
            distur_a, nodes_dict, h_dict, non_sv_nodes_with_blocked_np, curr_n_name_to_agent_list,
            blocked_nodes, full_corridor_check=True, avoid_curr_nodes=True, goals=goals_list, avoid_goals=True
        )
        distur_a.reset_alt_goal_node(distur_a_alter_goal_node, main_agent)
        agents = deque(agents)
        agents.remove(distur_a)
        agents.appendleft(distur_a)
        agents = list(agents)
        assert agents[0].get_goal_node().xy_name not in curr_n_name_to_agent_list
        return agents
    return agents


def all_update_return_roads(planned_agents: List[AlgCgarMapfAgent], iteration):
    # for the back-steps
    for agent in planned_agents:
        if len(agent.return_road) == 0:
            agent.return_road = deque([(agent.curr_node.xy_name, iteration-1, [], agent.curr_node)])
        agent_next_node = agent.path[iteration]
        if agent_next_node != agent.return_road[-1][3]:
            agent.return_road.append((agent_next_node.xy_name, iteration, [], agent_next_node))


def update_waiting_tables(
        agents_with_new_plan: List[AlgCgarMapfAgent], agents_to_return: List[AlgCgarMapfAgent],
        fs_to_a_dict: Dict[str, AlgCgarMapfAgent], to_assert: bool
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
                if agent_on_road.priority != 0 and agent_on_road != affected_agent:
                    assert agent_on_road in agents_with_new_plan
                    aor_n_name, aor_i, aor_a_list, aor_n = agent_on_road.return_road[-1]
                    assert aor_n_name == n_name
                    assert n == aor_n
                    aor_a_list.append(affected_agent.name)
                    affected_agent.add_to_wl(n, agent_on_road, aor_i, to_assert)


def calc_backward_road(
        backward_step_agents: List[AlgCgarMapfAgent],  # agents that are needed to be moved
        agents_with_new_plan: List[AlgCgarMapfAgent],  # agents that already planned
        agents_to_return: List[AlgCgarMapfAgent],  # all agents that need to return
        agents_dict: Dict[str, AlgCgarMapfAgent],
        from_n_to_a_dict: Dict[str, AlgCgarMapfAgent],
        future_captured_node_names: List[str],
        fs_to_a_dict: Dict[str, AlgCgarMapfAgent],
        to_config: Dict[str, Node],
        iteration: int, to_assert: bool = False
) -> None:
    # ------------------------------------------------------------------ #
    def update_data(given_a: AlgCgarMapfAgent, given_node: Node, to_pop: bool = False):
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

    # decide rest of config_to
    # open_list: Deque[AlgCgarMapfAgent] = deque(backward_step_agents[:])
    open_list: Deque[AlgCgarMapfAgent] = deque(agents_to_return[:])
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


def get_alter_goal_node(
        agent: AlgCgarMapfAgent, nodes_dict: Dict[str, Node], h_dict: dict,
        non_sv_nodes_with_blocked_np: np.ndarray, curr_n_name_to_agent_list: List[str], blocked_nodes: List[Node],
        full_corridor_check: bool = False, avoid_curr_nodes: bool = False,
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
        # not_in_alt_goal_nodes: bool = alt_node not in alt_goal_nodes
        alt_non_sv_np = non_sv_nodes_with_blocked_np[alt_node.x, alt_node.y]
        # alt_is_good, alt_message, i_error, info = is_enough_free_locations(
        #     agent.curr_node, alt_node, nodes_dict, h_dict, curr_nodes, alt_non_sv_np, blocked_nodes, full_corridor_check
        # )
        alt_is_good, alt_message, i_error, info = is_enough_free_locations(
            agent.curr_node, alt_node, nodes_dict, h_dict, curr_n_name_to_agent_list, alt_non_sv_np,
            blocked_nodes, full_corridor_check
        )
        # if not_curr_node and non_sv_in_main and not_in_alt_goal_nodes and alt_is_good:
        not_in_curr_nodes = True
        if avoid_curr_nodes:
            # not_in_curr_nodes = alt_node not in curr_nodes
            not_in_curr_nodes = alt_node.xy_name not in curr_n_name_to_agent_list
        not_in_goal_nodes = True
        if avoid_goals:
            not_in_goal_nodes = alt_node not in goals
        if not_curr_node and non_sv_in_main and alt_is_good and not_in_curr_nodes and not_in_goal_nodes:
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
        next_node_out_of_full_path = next_node not in closest_corridor
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


def all_others_if_at_goal_then_stay(
        agents: List[AlgCgarMapfAgent], main_agent: AlgCgarMapfAgent, iteration: int, to_assert: bool
):
    assert main_agent == agents[0]
    for agent in agents[1:]:
        if len(agent.path) - 1 == iteration - 1 and agent.path[-1] == agent.get_goal_node():
            agent.path.append(agent.path[-1])

# def get_alter_goal_node(
#         agent: AlgCgarMapfAgent, nodes_dict: Dict[str, Node], h_dict: dict, curr_nodes: List[Node],
#         non_sv_nodes_with_blocked_np: np.ndarray, blocked_nodes: List[Node],
#         full_corridor_check: bool = False, avoid_curr_nodes: bool = False,
#         goals_dict: List[Node] | None = None, avoid_goals: bool = False
# ) -> Node | None:
#     alter_goal_node = inner_get_alter_goal_node(
#         agent, nodes_dict, h_dict, curr_nodes, non_sv_nodes_with_blocked_np, blocked_nodes,
#         full_corridor_check=full_corridor_check, avoid_curr_nodes=avoid_curr_nodes,
#         goals_dict=goals_dict, avoid_goals=avoid_goals
#     )
#     if alter_goal_node is None:
#         return agent.goal_node
#     return alter_goal_node


# def liberate_goal_location(
#         main_agent: AlgCgarMapfAgent,
#         distur_a: AlgCgarMapfAgent,
#         agents: List[AlgCgarMapfAgent],
#         nodes: List[Node],
#         nodes_dict: Dict[str, Node],
#         h_dict: dict,
#         curr_nodes: List[Node],
#         non_sv_nodes_with_blocked_np: np.ndarray,
#         config_from: Dict[str, Node],
#         config_to: Dict[str, Node],
#         curr_n_name_to_agent_dict: Dict[str, AlgCgarMapfAgent],
#         curr_n_name_to_agent_list: List[str],
#         need_to_freeze_main_goal_node: bool,
#         iteration: int,
#         to_assert: bool):
#     if len(distur_a.path) - 1 >= iteration:
#         return [], []
#     if distur_a.alt_goal_node is not None:
#         assert distur_a.parent_of_goal_node_name == main_agent.name
#     blocked_nodes = get_blocked_nodes_for_ev(agents, iteration)
#     goals_list: List[Node] = [a.get_goal_node() for a in agents if a != distur_a]
#     blocked_nodes.extend(goals_list)
#     distur_a_alter_goal_node = get_alter_goal_node(distur_a, nodes_dict, h_dict, curr_nodes,
#                                                    non_sv_nodes_with_blocked_np, blocked_nodes)
#     distur_a.reset_alt_goal_node(distur_a_alter_goal_node, main_agent)
#
#     goals_dict = {a.name: a.get_goal_node() for a in agents}
#     regular_agent_decision(
#         distur_a,
#         agents,
#         nodes,
#         nodes_dict,
#         h_dict,
#         config_from,
#         config_to,
#         goals_dict,
#         curr_n_name_to_agent_dict,
#         curr_n_name_to_agent_list,
#         [],
#         non_sv_nodes_with_blocked_np,
#         need_to_freeze_main_goal_node,
#         curr_nodes,
#         iteration, to_assert)


# def reset_the_first_agent_if_not_achievable(
#         agents: List[AlgCgarMapfAgent],
#         nodes_dict: Dict[str, Node],
#         h_dict: dict,
#         curr_nodes: List[Node],
#         non_sv_nodes_with_blocked_np: np.ndarray,
#         iteration: int) -> List[AlgCgarMapfAgent]:
#     main_agent = agents[0]
#     given_goal_node = main_agent.get_goal_node()
#     a_non_sv_nodes_np = non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
#     backward_step_agents = [a for a in agents if len(a.return_road) > 0]
#     blocked_nodes = get_blocked_nodes_for_ev(agents, iteration, backward_step_agents=backward_step_agents)
#
#     is_good, message, i_error, info = is_enough_free_locations(
#         main_agent.curr_node, given_goal_node, nodes_dict, h_dict, curr_nodes, a_non_sv_nodes_np,
#         blocked_nodes, full_corridor_check=True)
#     if i_error in [5]:
#         print(f'\n{i_error=}, {message}', end='')
#         a_alter_goal_node = get_alter_goal_node(
#             main_agent, nodes_dict, h_dict, curr_nodes, non_sv_nodes_with_blocked_np,
#             blocked_nodes, full_corridor_check=True, avoid_curr_nodes=True)
#         main_agent.reset_alt_goal_node(a_alter_goal_node, main_agent)
#         print(f' ...alternative goal is: {a_alter_goal_node.xy_name}')
#     return agents


# def all_reset_return_roads(agents: List[AlgCgarMapfAgent], iteration: int):
#     for agent in agents:
#         agent.reset_return_road()