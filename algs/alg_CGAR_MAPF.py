from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt, build_vc_ec_from_configs


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
        for agent_name in obs['agents_names']:
            obs_agent = obs[agent_name]
            num = obs_agent.num
            start_node = self.nodes_dict[obs_agent.start_node_name]
            goal_node = self.nodes_dict[obs_agent.goal_node_name]
            new_agent = AlgCGARMAPFAgent(num=num, start_node=start_node, goal_node=goal_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.agents_num_dict[new_agent.num] = new_agent

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
        """
        # to render
        if to_render:
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            plot_rate = 0.001
            # plot_rate = 4

        # An example of While loop
        # iteration = 0
        # while not self.stop_condition():
        #     iteration += 1

        for agent in self.agents:
            pass




            # if to_assert:
            #     check_vc_ec_neic_iter(self.agents, iteration)
            #
            # # print + render
            # print(f'\r{'*' * 20} | [CGAR-MAPF] {iteration=} | {'*' * 20}', end='')
            # if to_render and iteration >= 0:
            #     # i_agent = self.agents_dict['agent_0']
            #     i_agent = self.agents[0]
            #     plot_info = {'img_np': self.img_np, 'agents': self.agents, 'i_agent': i_agent,
            #                  'non_sv_nodes_np': self.non_sv_nodes_np}
            #     plot_step_in_env(ax[0], plot_info)
            #     plot_return_paths(ax[1], plot_info)
            #     plt.pause(plot_rate)

        paths_dict = {a.name: a.path for a in self.agents}
        return True, paths_dict

    def stop_condition(self):
        for agent in self.agents:
            if agent.path[-1] != agent.goal_node:
                return False
        return True


@use_profiler(save_dir='../stats/alg_cgar_mapf.pstat')
def main():
    single_mapf_run(AlgCGARMAPF, is_SACGR=False)


if __name__ == '__main__':
    main()