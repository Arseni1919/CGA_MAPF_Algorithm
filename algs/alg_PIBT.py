import heapq

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF


def procedure_i_pibt[T](
        agent: T,
        agents: List[T],
        agents_dict: Dict[str, T],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        goals: Dict[str, Node],
        from_nodes: List[Node]
) -> bool:
    agent_name = agent.name
    h_goal_np: np.ndarray = h_dict[agent.next_goal_node.xy_name]

    # sort C in ascending order of dist(u, gi) where u ∈ C
    nei_nodes: List[Node] = [nodes_dict[n_name] for n_name in config_from[agent_name].neighbours]
    random.shuffle(nei_nodes)
    def get_nei_v(n):
        s_v = 0 if n not in from_nodes else 0.1
        return h_goal_np[n.x, n.y] + s_v
    nei_nodes.sort(key=get_nei_v)
    pass


def run_i_pibt[T](
        main_agent: T,
        agents: List[T],
        agents_dict: Dict[str, T],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        config_from: Dict[str, Node] | None = None,
        config_to: Dict[str, Node] | None = None,
        goals: Dict[str, Node] | None = None,
        from_nodes: List[Node] | None = None
) -> Dict[str, Node]:
    config_from: Dict[str, Node] = {agent.name: agent.curr_node for agent in agents} if config_from is None else config_from
    from_nodes: List[Node] = [agent.curr_node for agent in agents] if from_nodes is None else from_nodes
    goals: Dict[str, Node] = {agent.name: agent.goal_node for agent in agents} if goals is None else goals
    config_to: Dict[str, Node] = {} if config_to is None else config_to
    _ = procedure_i_pibt(main_agent, agents, agents_dict, nodes, nodes_dict, h_dict, config_from, config_to, goals, from_nodes)
    return config_to


def run_pibt[T](
        agents: List[T],
        agents_dict: Dict[str, T],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
) -> Dict[str, Node]:
    config_from: Dict[str, Node] = {agent.name: agent.curr_node for agent in agents}
    from_nodes = [agent.curr_node for agent in agents]
    heapq.heapify(from_nodes)
    goals: Dict[str, Node] = {agent.name: agent.goal_node for agent in agents}
    config_to: Dict[str, Node] = {}
    for agent in agents:
        if agent.name not in config_to:
            _ = run_i_pibt(agent, agents, agents_dict, nodes, nodes_dict, h_dict, config_from, config_to, goals, from_nodes)
    return config_to


@use_profiler(save_dir='../stats/alg_pibt.pstat')
def main():
    pass


if __name__ == '__main__':
    main()




