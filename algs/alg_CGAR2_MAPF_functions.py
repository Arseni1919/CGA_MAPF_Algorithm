from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt
from algs.alg_CGAR import align_all_paths, get_min_h_nei_node
from algs.alg_CGAR import build_corridor, find_ev_path, push_ev_agents, push_main_agent, build_corridor_from_nodes


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


def stay_where_you_are(main_agent: AlgCgar2MapfAgent):
    main_agent.path.append(main_agent.path[-1])


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def continuation_check_stage(
        main_agent: AlgCgar2MapfAgent,
        iteration: int,
        agents_to_return_dict: Dict[str, List[AlgCgar2MapfAgent]],
        agents_dict: Dict[str, AlgCgar2MapfAgent],
) -> Tuple[bool, dict]:
    # returns: to_resume: bool = True

    # if the agent has a plan
    if len(main_agent.path) - 1 >= iteration:
        return True, {'message': ''}

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
        message = 'swap' if the_order_swapped else ''
        return False, {'message': message}

    # Create blocked map




    return True, {'message': ''}




def calc_step_stage(
        main_agent: AlgCgar2MapfAgent
) -> None:
    pass


def return_agents_stage(
        main_agent: AlgCgar2MapfAgent
):
    pass










































