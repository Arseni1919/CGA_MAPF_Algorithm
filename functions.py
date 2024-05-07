import heapq

import numpy as np

from globals import *
from concurrent.futures import ThreadPoolExecutor


def save_results(**kwargs):
    algorithms = kwargs['algorithms']
    runs_per_n_agents = kwargs['runs_per_n_agents']
    img_dir = kwargs['img_dir']
    logs_dict = kwargs['logs_dict']
    file_dir = f'logs_for_plots/{datetime.now().strftime("%Y-%m-%d--%H-%M")}_ALGS-{len(algorithms)}_RUNS-{runs_per_n_agents}_MAP-{img_dir[:-4]}.json'
    # Serializing json
    json_object = json.dumps(logs_dict, indent=4)
    with open(file_dir, "w") as outfile:
        outfile.write(json_object)
    print(f'Results saved in: {file_dir}')
    return file_dir


def same_configs(config1: List[Tuple[int, int]], config2: List[Tuple[int, int]]) -> bool:
    for item1, item2 in zip(config1, config2):
        if item1 != item2:
            return False
    return True


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def pairwise_list[T](input_list: list) -> list[tuple[T, T]]:
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    result = list(map(lambda x: (x[0], x[1]), pairwise(input_list)))
    return result


def get_color(i):
    index_to_pick = i % len(color_names)
    return color_names[index_to_pick]


def check_stay_at_same_node(plan, the_node):
    for i_node in plan:
        if i_node.xy_name != the_node.xy_name:
            return False
    return True


def set_h_and_w(obj):
    if 'h' not in obj.params:
        return None, None
    else:
        return obj.params['h'], obj.params['w']


def set_pf_weight(obj):
    if 'pf_weight' not in obj.params:
        return 0
    else:
        return obj.params['pf_weight']


def use_profiler(save_dir):
    def decorator(func):
        def inner1(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            # getting the returned value
            returned_value = func(*args, **kwargs)
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.dump_stats(save_dir)
            # returning the value to the original frame
            return returned_value
        return inner1
    return decorator


def check_time_limit():
    def decorator(func):
        def inner1(*args, **kwargs):
            start_time = time.time()
            # getting the returned value
            returned_value = func(*args, **kwargs)
            end_time = time.time() - start_time
            if end_time > args[0].time_to_think_limit + 1:
                raise RuntimeError(f'[{args[0].alg_name}] crossed the time limit of {args[0].time_to_think_limit} s.')
            # returning the value to the original frame
            return returned_value
        return inner1
    return decorator


def set_seed(random_seed_bool, seed=1):
    if random_seed_bool:
        seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    print(f'[SEED]: --- {seed} ---')


def check_if_nei_pos(agents):
    for agent in agents:
        if agent.curr_node.xy_name not in agent.prev_node.neighbours:
            raise RuntimeError('wow wow wow! Not nei pos!')


def check_if_nei_pos_iter(agents, iteration):
    for agent in agents:
        if agent.path[iteration].xy_name not in agent.path[iteration - 1].neighbours:
            raise RuntimeError('wow wow wow! Not nei pos!')


def check_if_vc_iter(agents, iteration):
    for agent1, agent2 in combinations(agents, 2):
        if agent1.path[iteration] == agent2.path[iteration]:
            raise RuntimeError(f'vertex collision: {agent1.name} and {agent2.name} in {agent1.path[iteration].xy_name}')


def check_if_ec_iter(agents, iteration):
    for agent1, agent2 in combinations(agents, 2):
        prev_node1 = agent1.path[iteration - 1]
        curr_node1 = agent1.path[iteration]
        prev_node2 = agent2.path[iteration - 1]
        curr_node2 = agent2.path[iteration]
        edge1 = (prev_node1.x, prev_node1.y, curr_node1.x, curr_node1.y)
        edge2 = (curr_node2.x, curr_node2.y, prev_node2.x, prev_node2.y)
        if edge1 == edge2:
            raise RuntimeError(f'edge collision: {agent1.name} and {agent2.name} in {edge1}')


def check_vc_ec_neic_iter(agents: list, iteration: int) -> None:
    for a1, a2 in combinations(agents, 2):
        # vertex conf
        assert a1.path[iteration] != a2.path[iteration], f'[i: {iteration}] vertex conf: {a1.name}-{a2.name} in {a1.path[iteration].xy_name}'
        # edge conf
        prev_node1 = a1.path[max(0, iteration - 1)]
        curr_node1 = a1.path[iteration]
        prev_node2 = a2.path[max(0, iteration - 1)]
        curr_node2 = a2.path[iteration]
        edge1 = (prev_node1.x, prev_node1.y, curr_node1.x, curr_node1.y)
        edge2 = (curr_node2.x, curr_node2.y, prev_node2.x, prev_node2.y)
        assert edge1 != edge2, f'[i: {iteration}] edge collision: {a1.name}-{a2.name} in {edge1}'
        # nei conf
        assert a1.path[iteration].xy_name in a1.path[max(0, iteration - 1)].neighbours, f'[i: {iteration}] wow wow wow! Not nei pos!'
    assert agents[-1].path[iteration].xy_name in agents[-1].path[max(0, iteration - 1)].neighbours, f'[i: {iteration}] wow wow wow! Not nei pos!'


def check_paths(agents: list, from_iteration: int = 1) -> None:
    if len(agents) == 0:
        return
    max_len = max(map(lambda a: len(a.path), agents))
    for next_i in range(from_iteration, max_len):
        agents_to_check = [a for a in agents if next_i < len(a.path)]
        check_vc_ec_neic_iter(agents_to_check, next_i)
    print('\n ------------- Paths are good! -------------')


def plan_has_no_conf_with_vertex(plan, vertex):
    for plan_v in plan:
        if plan_v.x == vertex.x and plan_v.y == vertex.y:
            return False
    return True


def two_plans_have_no_confs(plan1, plan2):

    min_len = min(len(plan1), len(plan2))
    # assert len(plan1) == len(plan2)
    prev1 = None
    prev2 = None
    for i, (vertex1, vertex2) in enumerate(zip(plan1[:min_len], plan2[:min_len])):
        if vertex1.x == vertex2.x and vertex1.y == vertex2.y:
            return False
        if i > 0:
            # edge1 = (prev1.xy_name, vertex1.xy_name)
            # edge2 = (vertex2.xy_name, prev2.xy_name)
            # if (prev1.x, prev1.y, vertex1.x, vertex1.y) == (vertex2.x, vertex2.y, prev2.x, prev2.y):
            if prev1.x == vertex2.x and prev1.y == vertex2.y and vertex1.x == prev2.x and vertex1.y == prev2.y:
                return False
        prev1 = vertex1
        prev2 = vertex2
    return True


def two_plans_have_confs_at(plan1, plan2):

    min_len = min(len(plan1), len(plan2))
    assert len(plan1) == len(plan2)
    prev1 = None
    prev2 = None
    for i, (vertex1, vertex2) in enumerate(zip(plan1[:min_len], plan2[:min_len])):
        if vertex1.xy_name == vertex2.xy_name:
            return True, i
        if i > 0:
            # edge1 = (prev1.xy_name, vertex1.xy_name)
            # edge2 = (vertex2.xy_name, prev2.xy_name)
            if (prev1.xy_name, vertex1.xy_name) == (vertex2.xy_name, prev2.xy_name):
                return True, i
        prev1 = vertex1
        prev2 = vertex2
    return False, -1


def check_actions_if_vc(agents, actions):
    for agent1, agent2 in combinations(agents, 2):
        vertex1 = actions[agent1.name]
        vertex2 = actions[agent2.name]
        if vertex1 == vertex2:
            raise RuntimeError(f'vertex collision: {agent1.name} and {agent2.name} in {vertex1}')
            # print(f'\nvertex collision: {agent1.name} and {agent2.name} in {vertex1}')


def check_vc_ec_neic(agents):
    # check_if_nei_pos(self.agents) check_if_vc(self.agents) check_if_ec(self.agents)
    for a1, a2 in combinations(agents, 2):
        # vertex conf
        assert a1.curr_node != a2.curr_node, f'vertex conf: {a1.name}-{a2.name} in {a1.curr_node.xy_name}'
        # edge conf
        edge1 = (a1.prev_node.x, a1.prev_node.y, a1.curr_node.x, a1.curr_node.y)
        edge2 = (a2.curr_node.x, a2.curr_node.y, a2.prev_node.x, a2.prev_node.y)
        assert edge1 != edge2, f'edge collision: {a1.name}-{a2.name} in {edge1}'
        # nei conf
        assert a1.curr_node.xy_name in a1.prev_node.neighbours, 'wow wow wow! Not nei pos!'
    assert agents[-1].curr_node.xy_name in agents[-1].prev_node.neighbours, 'wow wow wow! Not nei pos!'


def check_if_vc(agents):
    for a1, a2 in combinations(agents, 2):
        assert a1.curr_node != a2.curr_node, f'vertex conf: {a1.name}-{a2.name} in {a1.curr_node.xy_name}'
        # if a1.curr_node == a2.curr_node:
        #     raise RuntimeError(f'vertex conf: {a1.name}-{a2.name} in {a1.curr_node.xy_name}')
            # print(f'\nvertex collision: {a1.name} and {a2.name} in {vertex1}')


def check_actions_if_ec(agents, actions):
    for agent1, agent2 in combinations(agents, 2):
        edge1 = (agent1.curr_node.xy_name, actions[agent1.name])
        edge2 = (actions[agent2.name], agent2.curr_node.xy_name)
        if edge1 == edge2:
            raise RuntimeError(f'edge collision: {agent1.name} and {agent2.name} in {edge1}')
            # print(f'\nedge collision: {agent1.name} and {agent2.name} in {edge1}')


def check_if_ec(agents):
    for a1, a2 in combinations(agents, 2):
        edge1 = (a1.prev_node.x, a1.prev_node.y, a1.curr_node.x, a1.curr_node.y)
        edge2 = (a2.curr_node.x, a2.curr_node.y, a2.prev_node.x, a2.prev_node.y)
        assert edge1 != edge2, f'edge collision: {a1.name}-{a2.name} in {edge1}'
        # if edge1 == edge2:
        #     raise RuntimeError(f'edge collision: {a1.name} and {a2.name} in {edge1}')
            # print(f'\nedge collision: {a1.name} and {a2.name} in {edge1}')


def create_sub_results(h_agents):
    # sub results
    sub_results = {}
    for agent in h_agents:
        # h_plan = agent.plan
        h_plan = [agent.curr_node]
        h_plan.extend(agent.plan)
        sub_results[agent.name] = h_plan
    # sub_results = {agent.name: agent.plan for agent in h_agents}
    return sub_results


def create_sds_sub_results(h_agents, nei_plans_dict):
    # sub results
    sub_results = {}
    for agent in h_agents:
        # h_plan = agent.plan
        h_plan = [agent.curr_node]
        h_plan.extend(nei_plans_dict[agent.name])
        sub_results[agent.name] = h_plan
    # sub_results = {agent.name: agent.plan for agent in h_agents}
    return sub_results


def build_constraints(nodes, other_paths):
    v_constr_dict = {node.xy_name: [] for node in nodes}
    e_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    xyt_problem = False

    for agent_name, path in other_paths.items():
        if len(path) > 0:
            final_node = path[-1]
            final_t = len(path) - 1
            perm_constr_dict[final_node.xy_name].append(final_t)
            perm_constr_dict[final_node.xy_name] = [max(perm_constr_dict[final_node.xy_name])]

            prev_node = path[0]
            for t, node in enumerate(path):
                # vertex
                v_constr_dict[f'{node.x}_{node.y}'].append(t)
                # edge
                if prev_node.xy_name != node.xy_name:
                    e_constr_dict[f'{prev_node.x}_{prev_node.y}'].append((node.x, node.y, t))
                    xyt_problem = True
                prev_node = node
    return v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem


def get_nei_nodes(curr_node, nei_r, nodes_dict):
    nei_nodes_dict = {}
    open_list = [curr_node]
    while len(open_list) > 0:
        i_node = open_list.pop()
        i_node_distance = euclidean_distance_nodes(curr_node, i_node)
        if i_node_distance <= nei_r:
            nei_nodes_dict[i_node.xy_name] = i_node
            for node_nei_name in i_node.neighbours:
                if node_nei_name not in nei_nodes_dict:
                    open_list.append(nodes_dict[node_nei_name])
    nei_nodes = list(nei_nodes_dict.values())
    return nei_nodes, nei_nodes_dict


def get_nei_nodes_times(curr_node, nei_r, nodes_dict):
    nei_nodes_dict = {}
    curr_node.t = 0
    open_list = [curr_node]
    while len(open_list) > 0:
        i_node = open_list.pop()
        i_node_distance = euclidean_distance_nodes(curr_node, i_node)
        if i_node_distance <= nei_r:
            nei_nodes_dict[i_node.xy_name] = i_node
            for node_nei_name in i_node.neighbours:
                if node_nei_name not in nei_nodes_dict:
                    node_nei = nodes_dict[node_nei_name]
                    node_nei.t = i_node.t + 1
                    open_list.append(node_nei)
    nei_nodes = list(nei_nodes_dict.values())
    return nei_nodes, nei_nodes_dict


def euclidean_distance_nodes(node1, node2):
    # p = [node1.x, node1.y]
    # q = [node2.x, node2.y]
    return math.dist([node1.x, node1.y], [node2.x, node2.y])
    # return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def manhattan_distance_nodes(node1, node2):
    return abs(node1.x-node2.x) + abs(node1.y-node2.y)

