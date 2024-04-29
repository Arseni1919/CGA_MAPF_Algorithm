import random
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *


class SimAgentMAPF:

    def __init__(self, num: int, start_node: Node):
        self.num = num
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.goal_node: Node | None = None
        self.path: List[Node] = []

    @property
    def name(self):
        return f'agent_{self.num}'


class SimEnvMAPF:
    def __init__(self, img_dir: str, is_SACGR=False, **kwargs):
        self.img_dir = img_dir
        self.is_SACGR = is_SACGR
        path_to_maps: str = kwargs['path_to_maps'] if 'path_to_maps' in kwargs else '../maps'
        path_to_heuristics: str = kwargs[
            'path_to_heuristics'] if 'path_to_heuristics' in kwargs else '../logs_for_heuristics'

        # for the map
        self.map_dim = get_dims_from_pic(img_dir=self.img_dir, path=path_to_maps)
        self.nodes, self.nodes_dict, self.img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
        self.h_dict = parallel_build_heuristic_for_entire_map(self.nodes, self.nodes_dict, self.map_dim,
                                                              img_dir=img_dir, path=path_to_heuristics)
        self.h_func = h_func_creator(self.h_dict)
        self.non_sv_nodes_np: np.ndarray = get_non_sv_nodes_np(self.nodes, self.nodes_dict, self.img_np, self.img_dir,
                                                               folder_dir='../logs_for_freedom_maps')
        self.agents: List[SimAgentMAPF] = []
        self.agents_dict: Dict[str, SimAgentMAPF] = {}
        self.start_nodes: List[Node] = []
        self.main_agent: SimAgentMAPF | None = None

    @property
    def n_agents(self):
        return len(self.agents)

    @property
    def start_nodes_names(self):
        return [n.xy_name for n in self.start_nodes]

    @property
    def goal_nodes_names(self):
        return [agent.goal_node.xy_name for agent in self.agents]

    @property
    def agents_names(self):
        return [a.name for a in self.agents]

    def reset(self, start_node_names: List[str] | None) -> Dict[str, Any]:
        self.start_nodes = [self.nodes_dict[snn] for snn in start_node_names]

        # create agents
        self._create_agents()

        # set goals
        if self.is_SACGR:
            self.main_agent = self.agents[0]
            occupied_nodes = [agent.start_node for agent in self.agents]
            free_nodes = [n for n in self.nodes if n not in occupied_nodes]
            goal_node: Node = random.choice(free_nodes)
            self.main_agent.goal_node = goal_node
            for agent in self.agents[1:]:
                agent.goal_node = agent.start_node

        else:
            self.main_agent = None
            goal_nodes = random.sample(self.nodes, self.n_agents)
            for agent, goal_node in zip(self.agents, goal_nodes):
                agent.goal_node = goal_node

        obs = self._get_obs()
        return obs

    def restart(self) -> Dict[str, Any]:
        prev_start_nodes = [a.start_node for a in self.agents]
        prev_goal_nodes = [a.goal_node for a in self.agents]

        self.start_nodes = prev_start_nodes
        # create agents
        self._create_agents()
        # set goals
        for agent, goal_node in zip(self.agents, prev_goal_nodes):
            agent.goal_node = goal_node

        obs = self._get_obs()
        return obs

    def _create_agents(self) -> None:
        # create agents
        self.agents: List[SimAgentMAPF] = []
        self.agents_dict: Dict[str, SimAgentMAPF] = {}
        for i, start_node in enumerate(self.start_nodes):
            new_agent = SimAgentMAPF(num=i, start_node=start_node)
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent

    def _get_obs(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {agent.name:
                                   AgentTupleMAPF(**{
                                       'num': agent.num,
                                       'start_node_name': agent.start_node.xy_name,
                                       'goal_node_name': agent.goal_node.xy_name,
                                   })
                               for agent in self.agents
                               }
        obs['start_nodes_names'] = self.start_nodes_names
        obs['goal_nodes_names'] = self.goal_nodes_names
        obs['agents_names'] = self.agents_names
        if self.is_SACGR:
            obs['main_agent_name'] = self.main_agent.name
        return obs

    def _get_metrics(self) -> dict:
        return {}


def main():
    N = 100
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-20.map'

    # problem creation
    # env = SimEnvMAPF(img_dir=img_dir)
    env = SimEnvMAPF(img_dir=img_dir, is_SACGR=True)
    start_nodes = random.sample(env.nodes, N)

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes])
    pprint(obs)


if __name__ == '__main__':
    main()
