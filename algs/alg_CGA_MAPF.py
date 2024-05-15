from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt
from algs.alg_CGAR import *
from algs.alg_CGAR_MAPF import is_enough_free_locations


def get_alter_goal_node[T](agent: T, nodes_dict, h_dict, curr_nodes, non_sv_nodes_with_blocked_np, blocked_nodes) -> Node | None:
    open_list = deque([agent.curr_node])
    closed_list_names = []
    main_goal_node: Node = agent.goal_node
    main_goal_non_sv_np = non_sv_nodes_with_blocked_np[main_goal_node.x, main_goal_node.y]
    while len(open_list) > 0:
        alt_node: Node = open_list.popleft()
        if main_goal_non_sv_np[alt_node.x, alt_node.y] and alt_node != agent.curr_node:
            alt_non_sv_np = non_sv_nodes_with_blocked_np[alt_node.x, alt_node.y]
            alt_is_good, alt_message = is_enough_free_locations(agent.curr_node, alt_node, nodes_dict, h_dict,
                                                                curr_nodes, alt_non_sv_np, blocked_nodes)
            main_is_good, main_message = is_enough_free_locations(alt_node, main_goal_node, nodes_dict, h_dict,
                                                                  curr_nodes, main_goal_non_sv_np, blocked_nodes)
            if alt_is_good and main_is_good:
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
    return None


def get_blocked_nodes[T](agents: T, iteration: int) -> List[Node]:
    blocked_nodes: List[Node] = []
    for agent in agents:
        # assert len(agent.path) - 1 == iteration - 1
        if len(agent.path) > iteration:
            for n in agent.path[iteration:]:
                heapq.heappush(blocked_nodes, n)
    # assert len(blocked_nodes_names) == 0
    return blocked_nodes


class AlgCgaMapfAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node, nodes: List[Node], nodes_dict: Dict[str, Node]):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.path: List[Node] = [start_node]
        self.first_arrived: bool = self.curr_node == self.goal_node
        if self.first_arrived:
            # Tuple: (0) node name, (1) iteration, (2) list of names of agents that will wait for you, (3) the node
            self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque(
                [(self.goal_node.xy_name, 0, [], self.goal_node)])
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
        return self.goal_node.xy_name

    @property
    def a_start_node_name(self):
        return self.start_node.xy_name

    def __eq__(self, other):
        return self.num == other.num

    def __hash__(self):
        return hash(self.num)

    def execute_forward_step(self, iteration: int) -> None:
        # execute the step
        self.prev_node = self.curr_node
        self.curr_node = self.path[iteration]
        assert self.prev_node.xy_name in self.curr_node.neighbours

        # if not self.first_arrived and self.curr_node == self.goal_node and len(self.path) - 1 == iteration:
        #     self.first_arrived = True
        #     self.return_road: Deque[Tuple[str, int, List[Self], Node]] = deque([(self.goal_node.xy_name, iteration, [], self.goal_node)])
        #     self.waiting_table: Dict[str, List[Tuple[str, int]]] = {n.xy_name: [] for n in self.nodes}
        #
        # # for the back-steps
        # if self.first_arrived:
        #     if self.curr_node != self.return_road[-1][3]:
        #         self.return_road.append((self.curr_node.xy_name, iteration, [], self.curr_node))

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


class AlgCgaMapf(AlgGeneric):
    name = 'CGA-MAPF'

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
        self.agents: List[AlgCgaMapfAgent] = []
        self.agents_dict: Dict[str, AlgCgaMapfAgent] = {}
        self.agents_num_dict: Dict[int, AlgCgaMapfAgent] = {}
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
    def curr_nodes(self):
        return [a.curr_node for a in self.agents]

    @property
    def curr_nodes_names(self):
        return [a.curr_node.xy_name for a in self.agents]

    @property
    def goal_nodes(self):
        return [a.goal_node for a in self.agents]

    @property
    def goal_nodes_names(self):
        return [a.goal_node.xy_name for a in self.agents]

    @property
    def n_solved(self) -> int:
        solved: List[AlgCgaMapfAgent] = [a for a in self.agents if a.goal_node == a.curr_node]
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
            new_agent = AlgCgaMapfAgent(num=num, start_node=start_node, goal_node=goal_node, nodes=self.nodes,
                                        nodes_dict=self.nodes_dict)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent
        self.n_agents = len(self.agents)

    def check_solvability(self) -> Tuple[bool, str]:
        # frankly, we need to check here the minimum number of free non-SV locations
        return True, 'good'

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[
        bool, Dict[str, List[Node]]]:
        """
        - while not everyone are at their goals:
            - calc_next_step()
            - execute forward steps
            (?) - execute backward steps
            - update priorities
        """

        # to render
        if to_render:
            plt.close()
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            plot_rate = 0.001
            # plot_rate = 2

        iteration = 0  # all at their start locations
        while not self.stop_condition():
            # current step iteration
            iteration += 1

            self.all_calc_next_steps(iteration, to_assert)
            fs_to_a_dict = self.all_execute_forward_steps(iteration, to_assert)
            # self.all_execute_backward_step(iteration, to_assert)
            self.update_priorities(fs_to_a_dict, to_assert)

            if to_assert:
                check_vc_ec_neic_iter(self.agents, iteration)

            # print + render
            print(f'\r{'*' * 20} | [CGA-MAPF] {iteration=} | solved: {self.n_solved}/{self.n_agents} | {'*' * 20}',
                  end='')
            if to_render and iteration >= 0:
                i_agent = self.agents[0]
                non_sv_nodes_np = self.non_sv_nodes_with_blocked_np[i_agent.goal_node.x, i_agent.goal_node.y]
                plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': i_agent,
                             'non_sv_nodes_np': non_sv_nodes_np}
                plot_step_in_env(ax[0], plot_info)
                # plot_return_paths(ax[1], plot_info)
                plt.pause(plot_rate)

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

    def all_calc_next_steps(self, iteration: int, to_assert: bool = False):

        # ---------------------------------------------- Preparations ---------------------------------------------- #
        # build blocked nodes
        blocked_nodes: List[Node] = get_blocked_nodes(self.agents, iteration)
        config_from: Dict[str, Node] = {agent.name: agent.curr_node for agent in self.agents}

        # ---------------------------------------------------------------------------------------------------------- #
        for i_priority, agent in enumerate(self.agents):
            # already planned
            if len(agent.path) - 1 >= iteration:
                continue
            given_goal_node = agent.goal_node
            a_non_sv_nodes_np = self.non_sv_nodes_with_blocked_np[given_goal_node.x, given_goal_node.y]
            if agent.prev_node == agent.curr_node:
                is_good, message = is_enough_free_locations(agent.curr_node, given_goal_node, self.nodes_dict, self.h_dict, self.curr_nodes, a_non_sv_nodes_np, blocked_nodes)
                if not is_good:
                    given_goal_node = get_alter_goal_node(agent, self.nodes_dict, self.h_dict, self.curr_nodes, self.non_sv_nodes_with_blocked_np, blocked_nodes)
                    # assert given_goal_node is not None
                    if given_goal_node is None:
                        continue
            a_next_node = get_min_h_nei_node(agent.curr_node, given_goal_node, self.nodes_dict, self.h_dict)
            if a_non_sv_nodes_np[a_next_node.x, a_next_node.y]:
                # calc single PIBT step
                self.calc_pibt_step(agent, given_goal_node, blocked_nodes, config_from, iteration, to_assert=to_assert)
            else:
                # elif i_priority == 0:
                # calc evacuation of agents from the corridor
                self.calc_ep_steps(agent, given_goal_node, config_from, a_non_sv_nodes_np, iteration,
                                   to_assert=to_assert)

            # update blocked nodes + check that there are new nodes added to the list
            blocked_nodes = get_blocked_nodes(self.agents, iteration)

        # if no plan - just stay
        for agent in self.agents:
            if len(agent.path) == iteration:
                agent.path.append(agent.path[-1])
        # ---------------------------------------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------------------------------------- #

    def calc_pibt_step(self, main_agent: AlgCgaMapfAgent, given_goal_node: Node, blocked_nodes: List[Node],
                       config_from: Dict[str, Node], iteration: int, to_assert: bool = False) -> Dict[str, Node]:
        # print(f'\n --- inside calc_pibt_step {iteration} --- ')
        assert len(main_agent.path) == iteration

        # Preps
        config_to = {}
        for agent in self.agents:
            if len(agent.path) - 1 >= iteration:
                config_to[agent.name] = agent.path[iteration]
        if to_assert:
            assert len(set(config_to.values())) == len(set(config_to.keys()))

        # Calc PIBT
        curr_blocked_nodes = blocked_nodes[:]
        curr_blocked_nodes.append(given_goal_node)
        config_to = run_i_pibt(
            main_agent=main_agent, agents=self.agents, nodes_dict=self.nodes_dict, h_dict=self.h_dict,
            config_from=config_from, config_to=config_to, blocked_nodes=curr_blocked_nodes,
            given_goal_node=given_goal_node)

        # Update paths
        for agent in self.agents:
            if len(agent.path) == iteration and agent.name in config_to:
                next_node = config_to[agent.name]
                agent.path.append(next_node)

        return config_to

    def calc_ep_steps(self, main_agent: AlgCgaMapfAgent, given_goal_node: Node, config_from: Dict[str, Node],
                      a_non_sv_nodes_np: np.ndarray, iteration: int, to_assert: bool = False) -> None:
        """
        - Build corridor
        - Build EP for ev-agents in the corridor
        - Evacuate ev-agents
        - Build the steps in the corridor to the main agent
        """
        assert len(main_agent.path) == iteration

        # Preps
        blocked_nodes = []
        for agent in self.agents:
            if len(agent.path) - 1 >= iteration:
                captured_nodes = agent.path[iteration-1:]
                blocked_nodes.extend(captured_nodes)
        curr_n_name_to_a_dict = {v.xy_name: self.agents_dict[k] for k, v in config_from.items()}
        curr_n_name_to_a_list: List[str] = list(curr_n_name_to_a_dict.keys())
        heapq.heapify(curr_n_name_to_a_list)

        # Calc
        corridor: List[Node] = build_corridor(main_agent, self.nodes_dict, self.h_dict, a_non_sv_nodes_np,
                                              given_goal_node=given_goal_node)
        assert corridor[0] == main_agent.path[-1]

        # if any of the corridor's nodes is blocked - just return
        if len([n for n in corridor[1:] if n in blocked_nodes]) > 0:
            return

        if corridor[-1] == given_goal_node and corridor[-1].xy_name in curr_n_name_to_a_dict:
            return

        # Find ev-agents
        ev_agents: List[AlgCgaMapfAgent] = []
        for node in corridor[1:]:
            if node.xy_name in curr_n_name_to_a_list:
                ev_agent = curr_n_name_to_a_dict[node.xy_name]
                assert ev_agent != main_agent
                ev_agents.append(ev_agent)

        # Build ev-paths (evacuation paths) for ev-agents in the corridor
        ev_paths_list: List[List[Node]] = []
        captured_free_nodes: List[Node] = []
        blocked_nodes.extend([main_agent.curr_node, given_goal_node])
        blocked_nodes = list(set(blocked_nodes))
        for ev_agent in ev_agents:
            ev_path, captured_free_node = find_ev_path(
                ev_agent.curr_node, corridor, self.nodes_dict, blocked_nodes, captured_free_nodes,
                curr_n_name_to_a_dict, curr_n_name_to_a_list
            )
            if ev_path is None:
                return
            captured_free_nodes.append(captured_free_node)
            ev_paths_list.append(ev_path)

        # Build steps for the ev-agents inside the ev-paths + extend paths
        moved_agents = []
        last_visit_dict = {n.xy_name: 0 for n in self.nodes}
        for i_ev_path, ev_path in enumerate(ev_paths_list):
            curr_n_name_to_a_dict: Dict[str, AlgCgaMapfAgent] = {a.path[-1].xy_name: a for a in self.agents}
            curr_n_name_to_a_list: List[str] = list(curr_n_name_to_a_dict.keys())
            max_len, assigned_agents = push_ev_agents(ev_path, curr_n_name_to_a_dict, curr_n_name_to_a_list,
                                                      moved_agents, self.nodes, main_agent, last_visit_dict,
                                                      iteration)
            assert main_agent not in assigned_agents
            # extend_other_paths(max_len, self.main_agent, self.agents)
            moved_agents.extend(assigned_agents)
            moved_agents = list(set(moved_agents))

        # Build the steps in the corridor to the main agent + extend the path
        push_main_agent(main_agent, corridor, moved_agents, iteration)

    def all_execute_forward_steps(self, iteration: int, to_assert: bool = False) -> Dict[str, AlgCgaMapfAgent]:
        # ps - previous step, fs - forward step
        # ps_to_a_dict: Dict[str, AlgCgaMapfAgent] = {a.curr_node.xy_name: a for a in self.agents}
        # a_name_to_ps_config_from: Dict[str, Node] = {a.name: a.curr_node for a in self.agents}
        fs_to_a_dict: Dict[str, AlgCgaMapfAgent] = {}
        # a_name_to_fs_config_to: Dict[str, Node] = {}
        for agent in self.agents:
            agent.execute_forward_step(iteration)
            fs_to_a_dict[agent.curr_node.xy_name] = agent
            # a_name_to_fs_config_to[agent.name] = agent.curr_node
        return fs_to_a_dict

    def update_priorities(self, fs_to_a_dict: Dict[str, AlgCgaMapfAgent], to_assert: bool = False) -> None:
        init_len = len(self.agents)
        unfinished: List[AlgCgaMapfAgent] = [a for a in self.agents if a.curr_node != a.goal_node]
        # random.shuffle(unfinished)
        goal_free_list: List[AlgCgaMapfAgent] = [a for a in unfinished if a.goal_node.xy_name not in fs_to_a_dict]
        not_goal_free_list: List[AlgCgaMapfAgent] = [a for a in unfinished if a.goal_node.xy_name in fs_to_a_dict]
        finished: List[AlgCgaMapfAgent] = [a for a in self.agents if a.curr_node == a.goal_node]
        random.shuffle(finished)
        # self.agents = [*unfinished, *finished]
        self.agents = [*goal_free_list, *not_goal_free_list, *finished]
        assert len(set(self.agents)) == init_len


@use_profiler(save_dir='../stats/alg_cga_mapf.pstat')
def main():
    # single_mapf_run(AlgCgaMapf, is_SACGR=True)
    single_mapf_run(AlgCgaMapf, is_SACGR=False)


if __name__ == '__main__':
    main()
