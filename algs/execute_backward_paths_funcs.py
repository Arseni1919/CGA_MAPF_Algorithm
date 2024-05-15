from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *


def get_a_visits_dict[T](agents: List[T], nodes: List[Node]) -> Dict[str, List[int]]:
    visits_dict: Dict[str, List[int]] = {n.xy_name: [] for n in nodes}
    for agent in agents:
        for n in agent.return_path_tuples:
            visits_dict[n.xy_name].append(agent.num)
    return visits_dict


def get_intersect_agents_from_a_visits_dict[T](agent: T, backward_step_agents: List[T], agents_num_dict: Dict[int, T], visits_dict: Dict[str, List[int]]) -> Tuple[List[T], List[str]]:
    intersect_agents: List[T] = []  # only from backward_step_agents
    all_group_nodes_names: List[str] = []  # not only from backward_step_agents
    backward_step_agents_nums: List[int] = [a.num for a in backward_step_agents]
    # open_list: Deque[int] = deque([agent.num])
    open_list: List[int] = [agent.num]
    closed_list: List[int] = []
    while len(open_list) > 0:
        next_agent_num = open_list.pop()
        next_agent = agents_num_dict[next_agent_num]
        rp_names: List[str] = []
        nei_agents: List[int] = []
        for n in next_agent.return_path_tuples:
            n_name = n.xy_name
            rp_names.append(n_name)
            nei_agents.extend(visits_dict[n_name])
        # rp_names: List[str] = [n.xy_name for n in next_agent.return_path_nodes]
        if next_agent_num in backward_step_agents_nums:
            intersect_agents.append(next_agent)
        all_group_nodes_names.extend(rp_names)

        for nei in set(nei_agents):
            if nei == next_agent_num:
                continue
            if nei in open_list:
                continue
            if nei in closed_list:
                continue
            # open_list.append(nei)
            heapq.heappush(open_list, nei)
        heapq.heappush(closed_list, next_agent_num)
    all_group_nodes_names = list(set(all_group_nodes_names))
    return intersect_agents, all_group_nodes_names


def cut_the_waiting[T](intersect_agents: List[T]) -> None:
    # return_path_nodes, iteration
    counts_list: List[int] = []
    agents_to_cut: List[T] = []
    for agent in intersect_agents:
        assert len(agent.return_path_tuples) != 0
        if len(agent.return_path_tuples) == 1:
            continue
        first_node = agent.return_path_tuples[-1]
        i_count = 0
        return_path_tuples_list = list(agent.return_path_tuples)
        for n in reversed(return_path_tuples_list[:-1]):
            if first_node == n:
                i_count += 1
                continue
            break
        if i_count == 0:
            return
        counts_list.append(i_count)
        agents_to_cut.append(agent)
    if len(agents_to_cut) == 0:
        return
    min_cut = min(counts_list)
    for agent in agents_to_cut:
        return_path_tuples_list = list(agent.return_path_tuples)
        agent.return_path_tuples = deque(return_path_tuples_list[:-min_cut])


def execute_backward_steps[T](backward_step_agents: List[T], future_captured_node_names: List[str], agents: List[T], agents_num_dict: Dict[int, T], main_agent: T, nodes: List[Node], iteration: int) -> None:
    for agent in backward_step_agents:
        assert len(agent.path) == iteration
        assert len(agent.return_path_tuples) != 0
    # for agent in backward_step_agents:
    #     agent.path.append(agent.path[-1])
    # return
    # intersect_graph: Dict[int, List[int]] = get_intersect_graph(agents, nodes)
    a_visits_dict: Dict[str, List[int]] = get_a_visits_dict(agents, nodes)
    for agent_1 in backward_step_agents:
        # if agent_1.num in [28, 29]:
        #     print(f'\nexecute_backward_steps {agent_1.name}: {agent_1.return_path_tuples_names}')
        # If you need to plan
        if len(agent_1.path) == iteration:
            # intersect_agents, all_nodes_a1_group = get_intersect_agents(agent_1, backward_step_agents, agents_num_dict, intersect_graph)
            intersect_agents, all_nodes_a1_group = get_intersect_agents_from_a_visits_dict(agent_1, backward_step_agents, agents_num_dict, a_visits_dict)
            # assert main_agent not in intersect_agents
            # If there are no possible collisions with the planned agents
            if set(all_nodes_a1_group).isdisjoint(future_captured_node_names):
                cut_the_waiting(intersect_agents)
                for i_agent in intersect_agents:
                    assert len(i_agent.return_path_tuples) != 0
                    assert i_agent.return_path_tuples[-1] == i_agent.path[-1]
                    if len(i_agent.return_path_tuples) == 1:
                        next_p_node = i_agent.return_path_tuples[-1]
                        assert next_p_node == i_agent.goal_node
                        i_agent.path.append(next_p_node)
                        i_agent.return_path_tuples = deque([next_p_node])
                        continue
                    curr_node = i_agent.return_path_tuples.pop()
                    next_p_node = i_agent.return_path_tuples[-1]
                    assert i_agent.path[-1].xy_name in next_p_node.neighbours
                    i_agent.path.append(next_p_node)
            else:
                for i_agent in intersect_agents:
                    i_agent.path.append(i_agent.path[-1])  # !!!
                    i_agent.return_path_tuples.append(i_agent.path[-1])

    # update paths + execute the step
    for agent in backward_step_agents:
        assert len(agent.path) == iteration + 1
        agent.prev_node = agent.curr_node
        agent.curr_node = agent.path[iteration]
        assert agent.curr_node == agent.return_path_tuples[-1]
        assert agent.prev_node.xy_name in agent.curr_node.neighbours
    return
