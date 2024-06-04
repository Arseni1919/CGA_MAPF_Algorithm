import heapq

import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt


def update_waiting_tables[T](agents: List[T], fs_to_a_dict: Dict[str, T], from_n_to_a_dict: Dict[str, T], iteration: int, to_assert: bool) -> None:
    # update waiting_table
    for affected_agent in agents:
        affected_agent_name = affected_agent.name
        if not affected_agent.first_arrived:
            continue
        assert len(affected_agent.return_road) != 0
        if len(affected_agent.return_road) == 1:
            assert affected_agent.curr_node == affected_agent.return_road[-1][3]
            continue
        for n_name, i, a_list, n in affected_agent.return_road:
            # inside next_n_name_to_a_dict
            if n_name in fs_to_a_dict:
                agent_on_road = fs_to_a_dict[n_name]
                agent_on_road_name = agent_on_road.name
                if agent_on_road.first_arrived and agent_on_road != affected_agent:
                    or_n_name, or_i, or_a_list, or_n = agent_on_road.return_road[-1]
                    assert or_n_name == n_name
                    assert n == or_n
                    or_a_list.append(affected_agent.name)
                    affected_agent.add_to_wl(n, agent_on_road, or_i, to_assert)


def execute_backward_road[T](from_n_to_a_dict: Dict[str, T], backward_step_agents: List[T], future_captured_node_names: List[str], fs_to_a_dict: Dict[str, T], to_config: Dict[str, Node],  agents: List[T], agents_dict: Dict[str, T], iteration: int, to_assert: bool = False) -> None:
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

    # decide rest of config_to
    open_list: Deque[T] = deque(agents[:])
    while len(open_list) > 0:
        next_agent = open_list.popleft()
        # already planned
        if next_agent.name in to_config:
            continue
        # did not arrived to the goal, therefore no need to return
        if not next_agent.first_arrived:
            update_data(next_agent, next_agent.curr_node)
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
        # TODO: check for circles - there will be never circles here = with the circles the algorithm will not work
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

    # execute + update a path
    for agent in backward_step_agents:
        assert len(agent.path) == iteration
        to_node = to_config[agent.name]
        agent.path.append(to_node)
        agent.prev_node = agent.curr_node
        agent.curr_node = to_node
        if to_assert:
            assert agent.curr_node == agent.return_road[-1][3]
            assert agent.prev_node.xy_name in agent.curr_node.neighbours
    return


def get_min_h_nei_node(curr_node: Node, goal_node: Node, nodes_dict: Dict[str, Node], h_dict: Dict[str, np.ndarray]) -> Node:
    nei_nodes = [nodes_dict[nn] for nn in curr_node.neighbours]
    goal_h_np: np.ndarray = h_dict[goal_node.xy_name]
    min_h_nei_node = min(nei_nodes, key=lambda n: goal_h_np[n.x, n.y])
    return min_h_nei_node


def build_corridor_from_nodes(curr_node: Node, goal_node: Node, nodes_dict: Dict[str, Node], h_dict: Dict[str, np.ndarray], non_sv_nodes_np: np.ndarray) -> List[Node]:
    main_next_node = get_min_h_nei_node(curr_node, goal_node, nodes_dict, h_dict)
    corridor: List[Node] = [curr_node, main_next_node]
    while non_sv_nodes_np[main_next_node.x, main_next_node.y] == 0 and main_next_node != goal_node:
        main_next_node = get_min_h_nei_node(main_next_node, goal_node, nodes_dict, h_dict)
        corridor.append(main_next_node)
    return corridor


def build_corridor[T](main_agent: T, nodes_dict: Dict[str, Node], h_dict: Dict[str, np.ndarray], non_sv_nodes_np: np.ndarray, given_goal_node: Node | None = None) -> List[Node]:
    main_goal_node = main_agent.goal_node
    if given_goal_node:
        main_goal_node = given_goal_node
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


def find_ev_path[T](curr_node: Node, corridor: List[Node], nodes_dict: Dict[str, Node], blocked_nodes: List[Node], captured_free_nodes: List[Node], curr_n_name_to_a_dict: Dict[str, T], curr_n_name_to_a_list: List[str]) -> Tuple[List[Node], Node] | Tuple[None, None]:
    open_list: Deque[Node] = deque([curr_node])
    open_names_list_heap = [curr_node.xy_name]
    heapq.heapify(open_names_list_heap)
    closed_names_list_heap = [n.xy_name for n in blocked_nodes]
    heapq.heapify(closed_names_list_heap)
    blocked_nodes_names = [n.xy_name for n in blocked_nodes]

    son_to_father_dict: Dict[str, Node | None] = {curr_node.xy_name: None}
    iteration: int = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.popleft()
        assert next_node not in blocked_nodes
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
    return None, None
    # raise RuntimeError('no way: cannot find an EV path')


def find_ev_path_m[T](curr_node: Node, corridor: List[Node], nodes_dict: Dict[str, Node], blocked_map: np.ndarray, captured_free_nodes: List[Node], curr_n_name_to_a_dict: Dict[str, T], curr_n_name_to_a_list: List[str]) -> Tuple[List[Node], Node] | Tuple[None, None]:
    open_list: Deque[Node] = deque([curr_node])
    open_names_list_heap = [curr_node.xy_name]
    heapq.heapify(open_names_list_heap)
    closed_names_list_heap = []
    # closed_names_list_heap = [n.xy_name for n in nodes_dict.values() if blocked_map[n.x, n.y]]
    # heapq.heapify(closed_names_list_heap)
    # blocked_nodes_names = [n.xy_name for n in blocked_nodes]

    son_to_father_dict: Dict[str, Node | None] = {curr_node.xy_name: None}
    iteration: int = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.popleft()
        assert blocked_map[next_node.x, next_node.y] == 0
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
            if blocked_map[nei_node.x, nei_node.y]:
                continue
            open_list.append(nei_node)
            heapq.heappush(open_names_list_heap, nei_name)
            son_to_father_dict[nei_name] = next_node
        heapq.heappush(closed_names_list_heap, next_node.xy_name)
    return None, None


def get_last_visit_dict[T](given_list: List[Node], given_agents: List[T], iteration: int) -> Dict[str, int]:
    last_visit_dict = {n.xy_name: 0 for n in given_list}
    for m_agent in given_agents:
        for i_n, n in enumerate(m_agent.path[iteration:]):
            if n in given_list:
                last_visit_dict[n.xy_name] = max(i_n + iteration, last_visit_dict[n.xy_name])
    return last_visit_dict


def update_last_visit_dict[T](last_visit_dict: Dict[str, int], given_agents: List[T]) -> None:
    for m_agent in given_agents:
        for i_n, n in enumerate(m_agent.path):
            last_visit_dict[n.xy_name] = max(i_n, last_visit_dict[n.xy_name])


def push_ev_agents[T](ev_path: List[Node], curr_n_name_to_a_dict: Dict[str, T], curr_n_name_to_a_list: List[str], moved_agents: List[T], nodes: List[Node], main_agent: T, last_visit_dict, iteration: int) -> Tuple[int, List[T]]:
    assert ev_path[0].xy_name in curr_n_name_to_a_list
    assert ev_path[-1].xy_name not in curr_n_name_to_a_list

    curr_moved_agents = moved_agents[:]

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
            assert i_agent != main_agent
            agents_to_assign.append(i_agent)
            locations_to_assign.append(n)
    locations_to_assign = locations_to_assign[1:]
    locations_to_assign.append(ev_path[-1])
    agents_to_assign.reverse()
    locations_to_assign.reverse()

    for a, final_n in zip(agents_to_assign, locations_to_assign):
        a_name = a.name
        new_path: List[Node] = []
        curr_node: Node = a.path[-1]
        while curr_node != final_n:
            next_node: Node = ev_chain_dict[curr_node.xy_name]
            next_n_last_visit = last_visit_dict[next_node.xy_name]
            while len(a.path) + len(new_path) <= next_n_last_visit:
                new_path.append(curr_node)
            assert next_node.xy_name in curr_node.neighbours
            new_path.append(next_node)
            curr_node = next_node
        assert a.path[-1].xy_name in new_path[0].neighbours
        a.path.extend(new_path)
        curr_moved_agents.append(a)
        update_last_visit_dict(last_visit_dict, [a])
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
    last_visit_dict = get_last_visit_dict(corridor, moved_agents, iteration)
    assert corridor[0] == main_agent.path[-1]
    prev_n = corridor[0]
    for c_n in corridor[1:]:
        c_n_last_visit = last_visit_dict[c_n.xy_name]
        while len(main_agent.path) <= c_n_last_visit:
            main_agent.path.append(prev_n)
        assert c_n.xy_name in prev_n.neighbours
        main_agent.path.append(c_n)
        prev_n = c_n


def align_all_paths[T](agents: T) -> int:
    max_len = max([len(a.path) for a in agents])
    for a in agents:
        while len(a.path) < max_len:
            a.path.append(a.path[-1])
    return max_len


class AlgCGARAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node, nodes: List[Node], nodes_dict: Dict[str, Node]):
        self.num = num
        self.priority = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.path: List[Node] = [start_node]
        self.first_arrived: bool = self.curr_node == self.goal_node
        if self.first_arrived:
            self.return_path_nodes: Deque[Node] = deque([self.goal_node])
            # Tuple: node name, iteration, list of names of agents that will wait for you, the node
            self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque([(self.goal_node.xy_name, 0, [], self.goal_node)])
        else:
            self.return_path_nodes: Deque[Node] = deque()
            self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque()
        self.trash_return_road: Deque[Tuple[str, int, List[Self], Node]] = deque()
        self.arrived: bool = self.first_arrived
        # Dictionary: node name -> List of (agent name, iteration)
        self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def return_path_nodes_names(self):
        return [n.xy_name for n in self.return_path_nodes]

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
        return self.goal_node.xy_name

    @property
    def a_start_node_name(self):
        return self.start_node.xy_name

    def get_goal_node(self):
        return self.goal_node

    def __eq__(self, other):
        return self.num == other.num

    def __hash__(self):
        return hash(self.num)

    def execute_forward_step(self, iteration: int) -> None:
        # execute the step
        self.prev_node = self.curr_node
        self.curr_node = self.path[iteration]
        assert self.prev_node.xy_name in self.curr_node.neighbours

        if not self.first_arrived and self.curr_node == self.goal_node and len(self.path) - 1 == iteration:
            self.first_arrived = True
            self.return_path_nodes: Deque[Node] = deque([self.goal_node])
            self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque([(self.goal_node.xy_name, iteration, [], self.goal_node)])
            self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}

        # for the back-steps
        if self.first_arrived:
            # self.return_path_nodes.append(self.curr_node)
            if len(self.return_path_nodes) == 1 and self.curr_node == self.goal_node:
                self.return_path_nodes = deque([self.curr_node])
                return
            self.return_path_nodes.append(self.curr_node)
            if self.curr_node != self.return_road[-1][3]:
                self.return_road.append((self.curr_node.xy_name, iteration, [], self.curr_node))

    def get_wl(self, node: Node):
        return self.waiting_table[node.xy_name]

    def get_wl_names(self, node: Node):
        return [tpl[0] for tpl in self.waiting_table[node.xy_name]]

    def add_to_wl(self, node: Node, agent_on_road: Self, iteration: int, to_assert: bool = False):
        # if to_assert:
        if (agent_on_road.name, iteration) not in self.waiting_table[node.xy_name]:
            self.waiting_table[node.xy_name].append((agent_on_road.name, iteration))

    def remove_from_wl(self, node: Node, agent_on_road: Self, iteration: int):
        remove_item = (agent_on_road.name, iteration)
        # assert remove_item in self.waiting_table[node.xy_name]
        self.waiting_table[node.xy_name] = [i for i in self.waiting_table[node.xy_name] if i != remove_item]

    def add_affected_agent(self, affected_agent: Self, iteration: int, n_name: str, n: Node, to_assert: bool = False):
        my_n_name, my_i, my_a_list, my_n = self.return_road[-1]
        my_a_list.append(affected_agent.name)
        # if to_assert:
        assert my_i == iteration
        assert my_n_name == n_name
        assert my_n == n


class AlgCGAR(AlgGeneric):
    name = 'CGAR'

    def __init__(self, env: SimEnvMAPF):
        super().__init__()
        # assert env.is_SACGR

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
        self.agents_num_dict: Dict[int, AlgCGARAgent] = {}
        self.main_agent: AlgCGARAgent | None = None

        # logs
        self.logs: dict | None = None

    @property
    def start_nodes(self):
        return [a.start_node for a in self.agents]

    @property
    def start_nodes_names(self):
        return [a.start_node.xy_name for a in self.agents]

    @property
    def curr_nodes(self):
        return [a.curr_node for a in self.agents]

    @property
    def goal_nodes(self):
        return [a.goal_node for a in self.agents]

    @property
    def goal_nodes_names(self):
        return [a.goal_node.xy_name for a in self.agents]

    def initialize_problem(self, obs: Dict[str, Any], non_sv_nodes_np: np.ndarray | None = None) -> None:
        # create agents
        self.agents = []
        self.agents_dict = {}
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgCGARAgent(num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes, nodes_dict=self.nodes_dict)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent

        self.main_agent = self.agents_dict[obs['main_agent_name']]
        if non_sv_nodes_np is not None:
            self.non_sv_nodes_np = non_sv_nodes_np
        else:
            blocked_nodes = [self.main_agent.goal_node]
            self.non_sv_nodes_np = get_non_sv_nodes_np(self.nodes, self.nodes_dict, self.img_np,
                                                       blocked_nodes=blocked_nodes)

        # for i_priority, agent in enumerate(self.agents):
        #     agent.future_rank = i_priority

    def check_solvability(self) -> Tuple[bool, str]:
        assert self.main_agent.goal_node != self.main_agent.start_node
        # if somebody stands in the goal location of the main agent
        if self.main_agent.goal_node in self.start_nodes:
            return False, 'goal_occupied'
        # if the current location is a non-SV
        if self.non_sv_nodes_np[self.main_agent.curr_node.x, self.main_agent.curr_node.y]:
            return True, 'good'
        # if there are enough free locations to allow the main agent to pass
        is_good, message, i_error, info = is_enough_free_locations_cgar(
            self.main_agent.curr_node, self.main_agent.goal_node, self.nodes_dict, self.h_dict, self.curr_nodes,
            self.non_sv_nodes_np, [], full_corridor_check=True)
        if not is_good:
            return False, message
        return True, message

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[bool, Dict[str, List[Node]]]:
        """
        - Block the goal vertex
        - While anyone is not in its goal:
            - If the plan is needed:
                - If the next vertex is non-SV:
                    - build PIBT step
                - Else:
                    - Build corridor
                    - Build EP for ev-agents in the corridor
                    - Evacuate ev-agents
                    - Build the steps in the corridor to the main agent
            - execute forward step
            - execute backward step
        - return
        """
        # is_good, message, i_error, info = is_enough_free_locations(
        #     self.main_agent.curr_node, self.main_agent.goal_node, self.nodes_dict, self.h_dict, self.curr_nodes,
        #     self.non_sv_nodes_np, [], full_corridor_check=True)
        # if not is_good:
        #     return False, {}

        if to_assert:
            set_lens = set([len(a.path) for a in self.agents])
            assert len(set_lens) == 1
            assert list(set_lens)[0] == 1
        # to render
        if to_render:
            plt.close()
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            plot_rate = 0.001
            # plot_rate = 2

        blocked_nodes = [self.main_agent.goal_node]
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
                    self.calc_pibt_step(iteration, blocked_nodes, to_assert=to_assert)
                else:
                    # calc evacuation of agents from the corridor
                    self.calc_ep_steps(iteration, blocked_nodes)

            future_captured_node_names: List[str] = []
            forward_step_agents: List[AlgCGARAgent] = []
            backward_step_agents: List[AlgCGARAgent] = []
            for agent in self.agents:
                if len(agent.path) > iteration:
                    forward_step_agents.append(agent)
                    for n in agent.path[iteration:]:
                        heapq.heappush(future_captured_node_names, n.xy_name)
                else:
                    backward_step_agents.append(agent)

            # execute the step
            # print('before execute_forward_step')
            ps_to_a_dict: Dict[str, AlgCGARAgent] = {a.curr_node.xy_name: a for a in self.agents}
            fs_to_a_dict: Dict[str, AlgCGARAgent] = {}
            to_config: Dict[str, Node] = {}
            for agent in forward_step_agents:
                agent.execute_forward_step(iteration)
                fs_to_a_dict[agent.curr_node.xy_name] = agent
                to_config[agent.name] = agent.curr_node
            # print('before execute_backward_steps')
            execute_backward_road(ps_to_a_dict, backward_step_agents, future_captured_node_names, fs_to_a_dict, to_config,  self.agents, self.agents_dict, iteration, to_assert=to_assert)
            # execute_backward_steps(backward_step_agents, future_captured_node_names, self.agents, self.agents_num_dict, self.main_agent, self.nodes, iteration)

            if to_assert:
                check_vc_ec_neic_iter(self.agents, iteration)

            # updates after the step execution
            iteration += 1

            # print + render
            print(f'\r{'*' * 20} | [CGAR] {iteration=} | {'*' * 20}', end='')
            if to_render and iteration >= 0:
                # i_agent = self.agents_dict['agent_0']
                i_agent = self.main_agent
                plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': i_agent, 'non_sv_nodes_np': self.non_sv_nodes_np}
                plot_step_in_env(ax[0], plot_info)
                plot_return_paths(ax[1], plot_info)
                plt.pause(plot_rate)
        # assert len(set([len(a.path) for a in self.agents])) == 1
        align_all_paths(self.agents)
        paths_dict = {a.name: a.path for a in self.agents}
        return True, paths_dict

    def calc_pibt_step(self, iteration: int, blocked_nodes: List[Node], to_assert: bool = False) -> None:
        # print(f'\n --- inside calc_pibt_step {iteration} --- ')
        if to_assert:
            assert len(self.main_agent.path) - 1 < iteration
        # Preps
        config_to = {}
        for agent in self.agents:
            if len(agent.path) - 1 >= iteration:
                config_to[agent.name] = agent.path[iteration]
        if to_assert:
            assert len(set(config_to.values())) == len(set(config_to.keys()))

        # Calc PIBT
        # print(f'{iteration=}')
        config_to = run_i_pibt(main_agent=self.main_agent, agents=self.agents, nodes_dict=self.nodes_dict, h_dict=self.h_dict, config_to=config_to, blocked_nodes=blocked_nodes)

        # Extend the paths
        for agent in self.agents:
            if agent.name in config_to:
                next_node = config_to[agent.name]
                agent.path.append(next_node)
        if to_assert:
            assert len(set(config_to.values())) == len(set(config_to.keys()))

    def calc_ep_steps(self, iteration: int, blocked_nodes: List[Node]) -> None:
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
                ev_agent = curr_n_name_to_a_dict[node.xy_name]
                if ev_agent != self.main_agent:
                    ev_agents.append(ev_agent)

        ev_paths_list: List[List[Node]] = []
        blocked_nodes.extend([self.main_agent.path[-1], self.main_agent.goal_node])
        blocked_nodes = list(set(blocked_nodes))
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
        last_visit_dict = {n.xy_name: 0 for n in self.nodes}
        for i_ev_path, ev_path in enumerate(ev_paths_list):
            curr_n_name_to_a_dict: Dict[str, AlgCGARAgent] = {a.path[-1].xy_name: a for a in self.agents}
            curr_n_name_to_a_list: List[str] = list(curr_n_name_to_a_dict.keys())
            max_len, assigned_agents = push_ev_agents(ev_path, curr_n_name_to_a_dict, curr_n_name_to_a_list, moved_agents, self.nodes, self.main_agent, last_visit_dict, iteration)
            assert self.main_agent not in assigned_agents
            # extend_other_paths(max_len, self.main_agent, self.agents)
            moved_agents.extend(assigned_agents)
            moved_agents = list(set(moved_agents))

        # Build the steps in the corridor to the main agent + extend the path
        push_main_agent(self.main_agent, corridor, moved_agents, iteration)

    def stop_condition(self):
        for agent in self.agents:
            if agent.path[-1] != agent.goal_node:
                return False
        return True


def is_enough_free_locations_cgar(
        curr_node: Node,
        goal_node: Node,
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        other_curr_nodes: List[Node],
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
        if next_node not in other_curr_nodes:
            return True, f'OK - next_node {next_node.xy_name} is a goal node and it is free', 0, {}
        return False, f'FAILED - next node {next_node.xy_name} is a goal node and is occupied', 1, {}

    closest_corridor: List[Node] = build_corridor_from_nodes(curr_node, goal_node, nodes_dict, h_dict, non_sv_nodes_np)
    if closest_corridor[-1] == goal_node and closest_corridor[-1] in other_curr_nodes:
        return False, f'FAILED - last corridor node {goal_node.xy_name} is a goal node and is occupied', 2, {
            'closest_corridor': [n.xy_name for n in closest_corridor]
        }

    if full_corridor_check:
        corridor_blocked_list = list(set(closest_corridor).intersection(blocked_nodes))
        if len(corridor_blocked_list) > 0:
            return False, f'FAILED - part of the corridor is blocked: {[n.xy_name for n in corridor_blocked_list]}', 3, {
                'closest_corridor': [n.xy_name for n in closest_corridor],
                'corridor_blocked_list': [n.xy_name for n in corridor_blocked_list]
            }

    # calc maximum required free nodes
    max_required_free_nodes = 0
    assert closest_corridor[0] == curr_node
    for n in closest_corridor[1:]:
        if n in other_curr_nodes:
            max_required_free_nodes += 1
    if max_required_free_nodes == 0:
        return True, f'OK - {max_required_free_nodes=}', 0, {
            'closest_corridor': [n.xy_name for n in closest_corridor]
        }

    # count available free locations
    free_count = 0
    touched_blocked_nodes = False
    touched_blocked_nodes_list: List[Node] = []
    while len(open_list) > 0:
        next_node = open_list.pop()
        open_list_names.remove(next_node.xy_name)
        # is_sv: bool = non_sv_nodes_np[next_node.x, next_node.y] == 0
        next_node_out_of_full_path = next_node not in full_path
        next_node_is_not_occupied = next_node not in other_curr_nodes
        if next_node_out_of_full_path and next_node_is_not_occupied:
            free_count += 1
            if free_count >= max_required_free_nodes:
                return True, f'OK - {free_count} free locations for {max_required_free_nodes=}', 0, {
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
    return False, 'FAILED - not_enough_free_nodes', error_num, {
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


@use_profiler(save_dir='../stats/alg_cgar.pstat')
def main():
    single_mapf_run(AlgCGAR, is_SACGR=True)


if __name__ == '__main__':
    main()

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


    # for agent in backward_step_agents:
    #     from_node: Node = config_from[agent.name]
    #     assert len(agent.return_nodes) != 0
    #     if len(agent.return_nodes) == 1:
    #         assert agent.return_nodes[0] == agent.goal_node
    #         config_to[agent.name] = from_node
    #         continue
    #     assert from_node == agent.return_nodes[-1]
    #     next_possible_node = agent.return_nodes[-2]
    #     next_possible_node_name = next_possible_node.xy_name
    #     if next_possible_node_name in future_captured_node_names:
    #         config_to[agent.name] = from_node
    #         continue
    #     next_possible_n_visitors = returns_dict[next_possible_node_name]
    #     assert agent.name in next_possible_n_visitors
    #     assert len(next_possible_n_visitors) != 0
    #     if len(next_possible_n_visitors) == 1:
    #         assert next_possible_n_visitors[0] == agent.name
    #         agent.return_nodes = agent.return_nodes[:-1]
    #         config_to[agent.name] = next_possible_node
    #         continue
    #     last_two_visitors_names = next_possible_n_visitors[-2:]
    #     if len(last_two_visitors_names) >= 2 and agent.name not in last_two_visitors_names:
    #         config_to[agent.name] = from_node
    #         continue
    #
    # config_to_n_names: List[str] = [n.xy_name for n in config_to.values()]
    # heapq.heapify(config_to_n_names)
    # config_from_n_names: List[str] = [n.xy_name for n in config_from.values()]
    # heapq.heapify(config_from_n_names)
    # from_n_to_a_name_dict: Dict[str, str] = {v.xy_name: k for k, v in config_from.items()}
    # assert len(from_n_to_a_name_dict) == len(config_from)
    #
    # for agent in backward_step_agents:
    #     if agent.name not in config_to:
    #         next_possible_node = agent.return_nodes[-2]
    #         next_possible_node_name = next_possible_node.xy_name
    #         config_to[agent.name] = next_possible_node
    #
    # there_are_collisions = True
    # while there_are_collisions:
    #     there_are_collisions = False
    #     for agent1, agent2 in combinations(backward_step_agents, 2):
    #         a1_from_node = config_from[agent1.name]
    #         a1_to_node = config_to[agent1.name]
    #         a2_from_node = config_from[agent2.name]
    #         a2_to_node = config_to[agent2.name]
    #         if a1_to_node == a2_to_node:
    #             config_to[agent1.name] = a1_from_node
    #             config_to[agent2.name] = a2_from_node
    #             there_are_collisions = True
    #             break
    #         if a1_from_node == a2_to_node and a1_to_node == a2_from_node:
    #             config_to[agent1.name] = a1_from_node
    #             config_to[agent2.name] = a2_from_node
    #             there_are_collisions = True
    #             break
    #
    # for agent in backward_step_agents:
    #     a_from_node = config_from[agent.name]
    #     a_to_node = config_to[agent.name]
    #     if a_from_node != a_to_node:
    #         next_n_visitors = returns_dict[a_from_node.xy_name]
    #         next_n_visitors.remove(agent.name)
    #         if len(agent.return_nodes) >= 2 and agent.return_nodes[-1] == a_from_node and agent.return_nodes[-2] == a_to_node:
    #             agent.return_nodes = agent.return_nodes[:-1]


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
    #             agent_to_push_name = curr_n_name_to_a_dict[next_possible_node_name]
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



    # for agent in backward_step_agents:
    #     # agent.path.append(agent.path[-1])
    #     # assert config_from[agent.name] == agent.return_nodes[-1]
    #     valid = procedure_i_pibt_back_step(
    #         agent, config_from, config_to, future_captured_node_names, agents_dict, from_n_to_a_name_dict)
    # to_n_to_a_name_dict: Dict[str, str] = {v.xy_name: k for k, v in config_to.items()}
    # assert len(to_n_to_a_name_dict) == len(config_to)

    # update paths + execute the step
    # for agent in backward_step_agents:
    #     assert agent.path[-1].xy_name in config_to[agent.name].neighbours
    #     next_node = config_to[agent.name]
    #     agent.path.append(next_node)
    #     agent.prev_node = agent.curr_node
    #     agent.curr_node = next_node
    # return




    # config_from: Dict[str, Node] = {a.name: a.path[-1] for a in backward_step_agents}
    # config_to: Dict[str, Node] = {}
    # from_n_to_a_name_dict: Dict[str, str] = {v.xy_name: k for k, v in config_from.items()}
    # assert len(from_n_to_a_name_dict) == len(config_from)



# def old_get_intersect_agents[T](agent: T, backward_step_agents: List[T], agents: List[T], intersect_graph: Dict[int, List[int]]) -> Tuple[List[T], List[str]]:
#     # find the full connected component and not only the direct neighbours
#     all_nodes_a1_group: List[str] = []
#     intersect_agents: List[T] = []
#     open_list: List[T] = [agent]
#     open_list_nums: List[str] = [agent.num]
#     closed_list_nums: List[str] = []
#     while len(open_list) > 0:
#         next_agent = open_list.pop()
#         open_list_nums.remove(next_agent.num)
#         next_l = [n.xy_name for i, n in next_agent.return_path_nodes]
#         all_nodes_a1_group.extend(next_l)
#         if next_agent in backward_step_agents:
#             intersect_agents.append(next_agent)
#         for agent_2 in agents:
#             if agent_2 == next_agent:
#                 continue
#             if agent_2.num in open_list_nums:
#                 continue
#             if agent_2.num in closed_list_nums:
#                 continue
#             a2_l = [n.xy_name for i, n in agent_2.return_path_nodes]
#             if not set(next_l).isdisjoint(a2_l):
#                 open_list.append(agent_2)
#                 heapq.heappush(open_list_nums, agent_2.num)
#         heapq.heappush(closed_list_nums, next_agent.num)
#     all_nodes_a1_group = list(set(all_nodes_a1_group))
#     assert len(intersect_agents) == len(set(intersect_agents))
#     return intersect_agents, all_nodes_a1_group

# def get_intersect_graph[T](agents: List[T], nodes: List[Node]) -> Dict[int, List[int]]:
#     # print('get_intersect_graph')
#     agents_to_check: List[T] = [a for a in agents if len(a.return_path_nodes) > 1]
#     intersect_graph: Dict[int, List[int]] = {a.num: [] for a in agents}
#     for agent1, agent2 in combinations(agents_to_check, 2):
#         if are_far_away(agent1, agent2):
#             continue
#         a1_l = [n.xy_name for i, n in agent1.return_path_nodes]
#         a2_l = [n.xy_name for i, n in agent2.return_path_nodes]
#         if not set(a1_l).isdisjoint(a2_l):
#             intersect_graph[agent1.num].append(agent2.num)
#             intersect_graph[agent2.num].append(agent1.num)
#         # found_connection: bool = False
#         # for i1, n1 in agent1.return_path_nodes:
#         #     for i2, n2 in agent2.return_path_nodes:
#         #         if n1 == n2:
#         #             intersect_graph[agent1.num].append(agent2.num)
#         #             intersect_graph[agent2.num].append(agent1.num)
#         #             found_connection = True
#         #             break
#         #     if found_connection:
#         #         break
#     return intersect_graph

# def are_far_away[T](agent1: T, agent2: T) -> bool:
#     # return False
#     len1 = len(agent1.return_path_nodes)
#     len2 = len(agent2.return_path_nodes)
#     return manhattan_distance_nodes(agent1.curr_node, agent2.curr_node) > len1 + len2

# def procedure_i_pibt_back_step[T](agent: T, config_from: Dict[str, Node], config_to: Dict[str, Node], future_captured_node_names: List[str], agents_dict: Dict[str, T], from_n_to_a_name_dict: Dict[str, str]) -> bool:
#     agent_name = agent.name
#     agent_curr_node = config_from[agent_name]
#     vc_set, ec_set = build_vc_ec_from_configs(config_from, config_to)
#     domain: List[Node] = get_list_of_next_return_nodes(agent, config_from)
#     for nei_node in domain:
#         # vc
#         if (nei_node.x, nei_node.y) in vc_set:
#             continue
#         # ec
#         if (nei_node.x, nei_node.y, agent_curr_node.x, agent_curr_node.y) in ec_set:
#             continue
#         # blocked
#         if nei_node.xy_name in future_captured_node_names:
#             continue
#
#         config_to[agent_name] = nei_node
#         if nei_node.xy_name in from_n_to_a_name_dict:
#             next_agent_name = from_n_to_a_name_dict[nei_node.xy_name]
#             next_agent = agents_dict[next_agent_name]
#             if agent != next_agent and next_agent_name not in config_to:
#                 next_is_valid = procedure_i_pibt_back_step(
#                     next_agent, config_from, config_to, future_captured_node_names, agents_dict, from_n_to_a_name_dict)
#                 if not next_is_valid:
#                     continue
#         if nei_node == agent.return_nodes[-1] and len(agent.return_nodes) > 1:
#             agent.return_nodes = agent.return_nodes[:-1]
#         return True
#     config_to[agent_name] = agent_curr_node
#     return False


# def get_list_of_next_return_nodes[T](agent: T, config_from: Dict[str, Node]) -> List[Node]:
#     agent_curr_node = config_from[agent.name]
#     assert len(agent.return_nodes) != 0
#     # assert agent.return_nodes[-1] == agent_curr_node
#     # you are at the goal location
#     if len(agent.return_nodes) == 1 and agent_curr_node == agent.goal_node:
#         return [agent_curr_node]
#     # you need to move
#     if agent.return_nodes[-1] == agent_curr_node:
#         agent.return_nodes = agent.return_nodes[:-1]
#     assert agent.return_nodes[-1] != agent_curr_node
#     next_possible_node = agent.return_nodes[-1]
#     # if you are first in the waiting list return both
#     pass
#     # if you are the second in the waiting list and somebody who was later (the first place) and he is on the spot -> return both
#     pass
#     # otherwise -> return curr_node
#     return [next_possible_node, agent_curr_node]


# def get_intersect_agents[T](agent: T, backward_step_agents: List[T], agents_num_dict: Dict[int, T], intersect_graph: Dict[int, List[int]]) -> Tuple[List[T], List[str]]:
#     # print('get_intersect_agents')
#     # find the full connected component and not only the direct neighbours
#     intersect_agents: List[T] = []
#     all_nodes_a1_group: List[str] = []
#     open_list: List[T] = [agent]
#     open_list_nums: List[int] = [agent.num]
#     closed_list_nums: List[int] = []
#     while len(open_list) > 0:
#         next_agent = open_list.pop()
#         open_list_nums.remove(next_agent.num)
#         next_l = [n.xy_name for i, n in next_agent.return_path_nodes]
#         all_nodes_a1_group.extend(next_l)
#         if next_agent in backward_step_agents:
#             intersect_agents.append(next_agent)
#         for nei_a_num in intersect_graph[next_agent.num]:
#             assert nei_a_num != next_agent.num
#             if nei_a_num in open_list_nums:
#                 continue
#             if nei_a_num in closed_list_nums:
#                 continue
#             nei_agent = agents_num_dict[nei_a_num]
#             open_list.append(nei_agent)
#             heapq.heappush(open_list_nums, nei_a_num)
#         heapq.heappush(closed_list_nums, next_agent.num)
#     all_nodes_a1_group = list(set(all_nodes_a1_group))
#     assert len(intersect_agents) == len(set(intersect_agents))
#     return intersect_agents, all_nodes_a1_group