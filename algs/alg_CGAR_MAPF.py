from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_CGAR import AlgCGAR, get_min_h_nei_node


def parallel_get_non_sv_nodes_np[T](agent: T, nodes: List[Node], nodes_dict: Dict[str, Node], img_np: np.ndarray, non_sv_nodes_np_dict: Dict[str, np.ndarray]) -> None:
    print(f'\n{agent.name} started...', end='')
    blocked_nodes = [agent.goal_node]
    non_sv_nodes_np = get_non_sv_nodes_np(nodes, nodes_dict, img_np, blocked_nodes=blocked_nodes)
    non_sv_nodes_np_dict[agent.name] = non_sv_nodes_np
    print(f'\n{agent.name} is finished.', end='')


def is_enough_free_locations(curr_node: Node, goal_node: Node, nodes_dict: Dict[str, Node], h_dict: Dict[str, np.ndarray], other_curr_nodes: List[Node], non_sv_nodes_np: np.ndarray) -> Tuple[bool, str]:
    next_node = get_min_h_nei_node(curr_node, goal_node, nodes_dict, h_dict)
    full_path: List[Node] = [next_node]
    open_list: List[Node] = [next_node]
    open_list_names: List[str] = [next_node.xy_name]
    closed_list: List[Node] = [curr_node, goal_node]
    closed_list_names: List[str] = [n.xy_name for n in closed_list]
    heapq.heapify(closed_list_names)

    # calc the biggest corridor
    sv_list = [0]
    sv_count = 0
    while next_node != goal_node:
        if not non_sv_nodes_np[next_node.x, next_node.y]:  # is SV
            sv_count += 1
        else:
            sv_list.append(sv_count)
            sv_count = 0
        next_node = get_min_h_nei_node(next_node, goal_node, nodes_dict, h_dict)
        full_path.append(next_node)
    max_corridor = max(sv_list)

    free_count = 0
    while len(open_list) > 0:
        next_node = open_list.pop()
        open_list_names.remove(next_node.xy_name)
        is_sv_and_in_full_path: bool = next_node in full_path and not non_sv_nodes_np[next_node.x, next_node.y]
        if not is_sv_and_in_full_path and next_node not in other_curr_nodes:
            free_count += 1
            if free_count >= max_corridor + 5:
                return True, 'good'
        for nei_name in next_node.neighbours:
            if nei_name == next_node.xy_name:
                continue
            if nei_name in closed_list_names:
                continue
            if nei_name in open_list_names:
                continue
            nei_node = nodes_dict[nei_name]
            open_list.append(nei_node)
            heapq.heappush(open_list_names, nei_name)
        heapq.heappush(closed_list_names, next_node.xy_name)

    return False, 'not_enough_free_nodes'


class AlgCGARMAPFAgent:
    def __init__(self, num: int, start_node: Node, goal_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node = goal_node
        self.path: List[Node] = [start_node]
        self.first_arrived: bool = self.curr_node == self.goal_node
        if self.first_arrived:
            self.return_path_tuples: Deque[Tuple[int, Node]] = deque([(0, self.goal_node)])
        else:
            self.return_path_tuples: Deque[Tuple[int, Node]] = deque()

    @property
    def name(self):
        return f'agent_{self.num}'

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    @property
    def return_path_tuples_names(self):
        return [(tpl[0], tpl[1].xy_name) for tpl in self.return_path_tuples]

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


class AlgCGARMAPF(AlgGeneric):

    name = 'CGAR-MAPF'

    def __init__(self, env: SimEnvMAPF):
        super().__init__()
        assert not env.is_SACGR

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
        self.agents: List[AlgCGARMAPFAgent] = []
        self.agents_dict: Dict[str, AlgCGARMAPFAgent] = {}
        self.agents_num_dict: Dict[int, AlgCGARMAPFAgent] = {}
        self.agents_open_list: deque[AlgCGARMAPFAgent] = deque()
        self.non_sv_nodes_np_dict: Dict[str, np.ndarray] = {}

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
    def agents_names(self):
        return [a.name for a in self.agents]

    def initialize_problem(self, obs: Dict[str, Any]) -> None:
        # create agents
        self.agents = []
        self.agents_dict = {}
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgCGARMAPFAgent(num=num, start_node=start_node, goal_node=goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent
        self.create_non_sv_nodes_np_dict()

    def create_non_sv_nodes_np_dict(self):
        print('\nStart of the create_non_sv_nodes_np_dict function..')
        self.non_sv_nodes_np_dict: Dict[str, np.ndarray] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            for agent in self.agents:
                executor.submit(parallel_get_non_sv_nodes_np, agent, self.nodes, self.nodes_dict, self.img_np, self.non_sv_nodes_np_dict)
        # for agent in self.agents:
        #     blocked_nodes = [agent.goal_node]
        #     non_sv_nodes_np = get_non_sv_nodes_np(self.nodes, self.nodes_dict, self.img_np, blocked_nodes=blocked_nodes)
        #     self.non_sv_nodes_np_dict[agent.name] = non_sv_nodes_np
        #     print(f'\n{agent.name} is finished.', end='')
        print('\nCreate_non_sv_nodes_np_dict is finished.')

    def check_solvability(self) -> Tuple[bool, str]:
        return True, 'good'

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[bool, Dict[str, List[Node]]]:
        """
        - For every agent a:
            - If there is disturbing agent distur_a in the goal vertex:
                - free_v <- Find closest achievable non-SV for distur_a
                - CGAR(distur_a, g=free_v)
            - If not achievable:
                - free_v <- Find closest achievable non-SV for the main agent
                - CGAR(main_a, g=free_v)
            - Execute CGAR(a)
            - Compress paths with the previous round
        """
        # to render
        if to_render:
            # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            plot_rate = 0.001
            # plot_rate = 4

        self.order_agents(init=True, to_assert=to_assert)
        iteration: int = 0
        while len(self.agents_open_list) > 0:
            iteration += 1
            # if iteration >= 55:
            #     to_render = True
            next_agent = self.agents_open_list.popleft()

            # If there is disturbing agent distur_a in the goal vertex
            for i_agent in self.agents:
                if i_agent.curr_node == next_agent.goal_node:
                    print(f'\nneed to move {i_agent.name}')
                    new_goal_node = self.get_nearest_goal_node(i_agent=i_agent)
                    # Execute CGAR(a) + compress
                    paths_dict = self.execute_cgar(i_agent, new_goal_node, max_time, to_assert, to_render)
                    self.compress(paths_dict, to_assert=to_assert)

            # if the current location is not good
            blocked_nodes: List[Node] = []
            alt_start_node: Node = next_agent.curr_node
            is_good, message = is_enough_free_locations(alt_start_node, next_agent.goal_node, self.nodes_dict, self.h_dict, self.curr_nodes, self.non_sv_nodes_np)
            if not is_good:
                while not is_good:
                    print(f'\n{message} -- changing alt_start_node')
                    blocked_nodes.append(alt_start_node)
                    alt_start_node = self.get_nearest_goal_node(i_agent=next_agent, blocked_nodes=blocked_nodes)
                    is_good, message = is_enough_free_locations(alt_start_node, next_agent.goal_node, self.nodes_dict,
                                                                self.h_dict, self.curr_nodes, self.non_sv_nodes_np)
                # Execute CGAR(a) + compress
                paths_dict = self.execute_cgar(next_agent, alt_start_node, max_time, to_assert, to_render)
                self.compress(paths_dict, to_assert=to_assert)

            # Execute CGAR(a) + compress
            print(f'\nregular execute')
            paths_dict = self.execute_cgar(next_agent, next_agent.goal_node, max_time, to_assert, to_render)
            self.compress(paths_dict, to_assert=to_assert)

            self.order_agents(prev_agent=next_agent, to_assert=to_assert)

            # if to_assert:
            #     check_vc_ec_neic_iter(self.agents, iteration)
            #
            # print + render
            print(f'\n{'*' * 20} | [CGAR-MAPF] finished: {iteration}/{len(self.agents)} | {'*' * 20}')
            if to_render:
                pass
                # i_agent = self.agents_dict['agent_0']
                # plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': next_agent,
                #              'non_sv_nodes_np': self.non_sv_nodes_np}
                # plot_step_in_env(ax[0], plot_info)
                # plot_return_paths(ax[1], plot_info)
                # plt.pause(plot_rate)

        paths_dict = {a.name: a.path for a in self.agents}
        solved = self.stop_condition()
        if to_assert:
            assert solved
            check_paths(self.agents)
        return solved, paths_dict

    def stop_condition(self):
        for agent in self.agents:
            if agent.path[-1] != agent.goal_node:
                return False
        return True

    def order_agents(self, init: bool = False, prev_agent: AlgCGARMAPFAgent | None = None, to_assert: bool = False) -> None:
        if init:
            self.agents_open_list = deque(self.agents)
        if to_assert:
            assert len(set([len(a.path) for a in self.agents])) == 1
            for agent in self.agents:
                assert agent.path[-1] == agent.curr_node
        goal_free_list, goal_not_free_list = [], []
        for agent in self.agents_open_list:
            if agent.goal_node in self.curr_nodes:
                goal_not_free_list.append(agent)
            else:
                goal_free_list.append(agent)
        if prev_agent:
            # the far away the better
            goal_free_list.sort(key=lambda a: self.h_dict[prev_agent.curr_node.xy_name][a.curr_node.x, a.curr_node.y] + self.h_dict[prev_agent.curr_node.xy_name][a.goal_node.x, a.goal_node.y], reverse=True)
            # goal_free_list.sort(key=lambda a: dist_heuristic(prev_agent.curr_node, a.curr_node), reverse=True)
        self.agents_open_list = deque([*goal_free_list, *goal_not_free_list])

    def build_obs(self, main_agent: AlgCGARMAPFAgent, goal_node) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}
        goal_nodes_names = []
        for agent in self.agents:
            if agent == main_agent:
                obs[agent.name] = AgentTupleMAPF(**{
                                       'num': agent.num,
                                       'start_node_name': agent.curr_node.xy_name,
                                       'goal_node_name': goal_node.xy_name,
                                   })
                goal_nodes_names.append(goal_node.xy_name)
            else:
                obs[agent.name] = AgentTupleMAPF(**{
                    'num': agent.num,
                    'start_node_name': agent.curr_node.xy_name,
                    'goal_node_name': agent.curr_node.xy_name,
                })
                goal_nodes_names.append(agent.curr_node.xy_name)
        obs['start_nodes_names'] = self.curr_nodes_names
        obs['goal_nodes_names'] = goal_nodes_names
        obs['agents_names'] = self.agents_names
        obs['main_agent_name'] = main_agent.name
        return obs

    def execute_cgar(self, main_agent: AlgCGARMAPFAgent, goal_node: Node, max_time: int, to_assert: bool = True, to_render: bool = False) -> Dict[str, List[Node]]:
        # Execute CGAR(a)
        obs = self.build_obs(main_agent=main_agent, goal_node=goal_node)
        i_alg_cgar = AlgCGAR(env=self.env)
        i_alg_cgar.initialize_problem(obs=obs, non_sv_nodes_np=self.non_sv_nodes_np_dict[main_agent.name])
        solved, paths_dict = i_alg_cgar.solve(max_time=max_time, to_assert=to_assert, to_render=to_render)
        return paths_dict

    def compress(self, paths_dict: Dict[str, List[Node]], to_assert: bool = False) -> None:
        if to_assert:
            assert len(set([len(a.path) for a in self.agents])) == 1
            assert len(set([len(path) for a_name, path in paths_dict.items()])) == 1
        # for a in self.agents:
        #     a.path.extend(paths_dict[a.name])
        #     a.curr_node = a.path[-1]
        #     if len(a.path) > 1:
        #         a.prev_node = a.path[-2]
        # return

        # all moved
        max_new_len = len(paths_dict[self.agents[0].name])
        max_old_len = len(self.agents[0].path)
        if max_old_len <= max_new_len:
            for a in self.agents:
                a.path.extend(paths_dict[a.name])
                a.curr_node = a.path[-1]
                if len(a.path) > 1:
                    a.prev_node = a.path[-2]
            return

        all_now_moved: List[AlgCGARMAPFAgent] = []
        for agent in self.agents:
            for n1, n2 in pairwise(paths_dict[agent.name]):
                if n1 != n2:
                    all_now_moved.append(agent)
                    break

        nodes_names_of_now_moved: List[str] = []
        for m_agent in all_now_moved:
            nodes_names_of_now_moved.extend([n.xy_name for n in paths_dict[m_agent.name]])

        all_prev_moved: List[AlgCGARMAPFAgent] = []
        for agent in self.agents:
            for n1, n2 in pairwise(agent.path[-max_new_len - 1:]):
                if n1 != n2:
                    all_prev_moved.append(agent)
                    break
        # [n.xy_name for n in m_agent.path[-max_new_len:]]

        nodes_names_of_all_prev: List[str] = []
        for prev_agent in all_prev_moved:
            nodes_names_of_all_prev.extend([n.xy_name for n in prev_agent.path[-max_new_len:]])

        can_compress: bool = set(nodes_names_of_all_prev).isdisjoint(nodes_names_of_now_moved)

        if can_compress:
            for m_agent in all_now_moved:
                cut_part = [n.xy_name for n in m_agent.path[-max_new_len:]]
                new_part = [n.xy_name for n in paths_dict[m_agent.name]]


                assert m_agent.path[-1] == paths_dict[m_agent.name][0]
                m_agent.path = m_agent.path[:-max_new_len]
                assert m_agent.path[-1] == paths_dict[m_agent.name][0]
                m_agent.path.extend(paths_dict[m_agent.name])
                assert len(m_agent.path) == max_old_len
        else:
            for agent in self.agents:
                agent.path.extend(paths_dict[agent.name])

        for agent in self.agents:
            agent.curr_node = agent.path[-1]
            if len(agent.path) > 1:
                agent.prev_node = agent.path[-2]

        if to_assert:
            assert len(set([len(a.path) for a in self.agents])) == 1

    def get_nearest_goal_node(self, i_agent: AlgCGARMAPFAgent, blocked_nodes: List[Node] | None = None) -> Node:
        curr_nodes_names = self.curr_nodes_names
        open_list: Deque[Node] = deque([i_agent.curr_node])
        open_names_list_heap = [f'{i_agent.curr_node.xy_name}']
        closed_names_list_heap = []
        if blocked_nodes is None:
            blocked_nodes = []

        while len(open_list) > 0:
            next_n = open_list.pop()
            open_names_list_heap.remove(next_n.xy_name)
            if self.non_sv_nodes_np[next_n.x, next_n.y] and next_n.xy_name not in curr_nodes_names and next_n not in blocked_nodes:
                is_good, message = is_enough_free_locations(i_agent.curr_node, next_n, self.nodes_dict, self.h_dict, self.curr_nodes, self.non_sv_nodes_np)
                if is_good:
                    return next_n
            for nei_name in next_n.neighbours:
                if nei_name == next_n.xy_name:
                    continue
                if nei_name in open_names_list_heap:
                    continue
                if nei_name in closed_names_list_heap:
                    continue
                open_list.append(self.nodes_dict[nei_name])
                heapq.heappush(open_names_list_heap, nei_name)
            heapq.heappush(closed_names_list_heap, next_n.xy_name)
        raise RuntimeError('strange...')


@use_profiler(save_dir='../stats/alg_cgar_mapf.pstat')
def main():
    single_mapf_run(AlgCGARMAPF, is_SACGR=False)


if __name__ == '__main__':
    main()


    # def compress(self, paths_dict: Dict[str, List[Node]], to_assert: bool = False) -> None:
    #     bridges: Dict[str, int] = {a.name: 0 for a in self.agents}
    #
    #     # cut the old paths
    #     max_old_len = len(self.agents[0].path)
    #     old_paths_dict: Dict[str, List[Node]] = {}
    #     for agent in self.agents:
    #         to_cut = 0
    #         for node1, node2 in pairwise(reversed(agent.path)):
    #             if node1 == node2:
    #                 to_cut += 1
    #                 continue
    #             break
    #         old_paths_dict[agent.name] = agent.path[: len(agent.path) - to_cut]
    #         bridges[agent.name] += to_cut
    #
    #     # cut the new paths
    #     max_new_len = len(paths_dict[self.agents[0].name])
    #     new_paths_dict: Dict[str, List[Node]] = {}
    #     for agent in self.agents:
    #         to_cut = 0
    #         for node1, node2 in pairwise(agent.path):
    #             if node1 == node2:
    #                 to_cut += 1
    #                 continue
    #             break
    #         new_paths_dict[agent.name] = agent.path[to_cut:]
    #         bridges[agent.name] += to_cut
    #
    #     # cut the bridges
    #     min_cut = min(bridges.values())
    #     for agent in self.agents:
    #         bridges[agent.name] -= min_cut
    #     assert min(bridges.values()) == 0
    #
    #     # concat paths
    #     for agent in self.agents:
    #         p1 = old_paths_dict[agent.name]
    #         p2 = new_paths_dict[agent.name]
    #         assert p1[-1] == p2[0]
    #         bridge_num = bridges[agent.name]
    #         bridge = [p1[-1] for _ in range(bridge_num)]
    #         full_path = [*p1, *bridge, *p2]
    #         agent.path = full_path
    #         agent.curr_node = agent.path[-1]
    #         if len(agent.path) > 1:
    #             agent.prev_node = agent.path[-2]
    #     if to_assert:
    #         assert len(set([len(a.path) for a in self.agents])) == 1