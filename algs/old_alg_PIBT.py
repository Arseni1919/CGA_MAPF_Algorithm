from globals import *
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *

from environments.env_SACG_LMAPF import SimEnvLMAPF
from algs.alg_gen_cor_v1 import copy_nodes
from algs.alg_clean_corridor import *
from create_animation import do_the_animation
from algs.params import *
from algs.alg_CGA import ALgCGA, AlgCGAAgent


def get_vc_ac_sets(planned_agents: List[AlgCGAAgent], next_iteration: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    vc_set, ec_set = [], []
    for agent in planned_agents:
        curr_node = agent.path[next_iteration - 1]
        next_node = agent.path[next_iteration]
        vc_set.append((next_node.x, next_node.y))
        ec_set.append((curr_node.x, curr_node.y, next_node.x, next_node.y))
    heapq.heapify(vc_set)
    heapq.heapify(ec_set)
    return vc_set, ec_set


def pibt_func(agent: AlgCGAAgent,
              h_dict: Dict[str, np.ndarray],
              nodes_dict: Dict[str, Node],
              vc_set: List[Tuple[int, int]],
              ec_set: List[Tuple[int, int, int, int]],
              node_name_to_agent_dict: Dict[str, AlgCGAAgent],
              node_name_to_agent_list: List[str],
              next_iteration: int,
              logs: dict | None) -> bool:
    # print(f'\rin PIBT func {agent.name}', end='')
    h_goal_np: np.ndarray = h_dict[agent.next_goal_node.xy_name]

    # sort C in ascending order of dist(u, gi) where u ∈ C
    nei_nodes_names = agent.curr_node.neighbours[:]
    nei_nodes: List[Node] = list(map(lambda nnn: nodes_dict[nnn], nei_nodes_names))
    random.shuffle(nei_nodes)
    nei_nodes.sort(key=lambda n: h_goal_np[n.x, n.y])

    # if next_iteration == 26 and agent.num in [41, 534]:
    # if next_iteration == 26 and agent.num in [41]:
    #     print(f'\n{agent.name=}')

    for nei_node in nei_nodes:
        if logs is not None:
            logs['expanded_nodes'] += 1
        # if collisions in Q to supposing Qto[i] = v then continue
        # vc
        if (nei_node.x, nei_node.y) in vc_set:
            continue
        # ec
        if (nei_node.x, nei_node.y, agent.curr_node.x, agent.curr_node.y) in ec_set:
            continue

        agent.path = agent.path[:next_iteration]
        # assert len(agent.path[next_iteration:]) == 0
        agent.path.append(nei_node)
        # assert len(agent.path[next_iteration:]) == 1
        # assert (nei_node.x, nei_node.y) not in vc_set
        heapq.heappush(vc_set, (nei_node.x, nei_node.y))
        # assert (agent.curr_node.x, agent.curr_node.y, nei_node.x, nei_node.y) not in ec_set
        heapq.heappush(ec_set, (agent.curr_node.x, agent.curr_node.y, nei_node.x, nei_node.y))

        if nei_node.xy_name in node_name_to_agent_list:
            next_agent = node_name_to_agent_dict[nei_node.xy_name]
            if agent != next_agent and len(next_agent.path[next_iteration:]) == 0:
                next_is_good = pibt_func(
                    next_agent,
                    h_dict,
                    nodes_dict,
                    vc_set,
                    ec_set,
                    node_name_to_agent_dict,
                    node_name_to_agent_list,
                    next_iteration,
                    logs
                )
                if not next_is_good:
                    agent.path = agent.path[:next_iteration]
                    # assert len(agent.path[next_iteration:]) == 0
                    continue
        return True

    agent.path = agent.path[:next_iteration]
    # assert len(agent.path[next_iteration:]) == 0
    next_node = agent.path[-1]
    agent.path.append(next_node)
    # assert len(agent.path[next_iteration:]) == 1
    if (next_node.x, next_node.y) not in vc_set:
        heapq.heappush(vc_set, (next_node.x, next_node.y))
    return False


class AlgPIBT(ALgCGA):
    name = 'PIBT'

    def __init__(self, env: SimEnvLMAPF, **kwargs):
        super().__init__(env, **kwargs)
        self.freedom_nodes_np: np.ndarray = np.ones(self.img_np.shape)

    def _calc_next_steps(self) -> None:
        start_time = time.time()
        for agent in self.agents:

            # already planned
            if len(agent.path) - 1 == self.next_iteration:
                continue

            # no target
            if agent.curr_node == agent.next_goal_node:
                continue

            planned_agents = list(filter(lambda a: len(a.path[self.next_iteration:]) > 0, self.agents))
            vc_set, ec_set = get_vc_ac_sets(planned_agents, self.next_iteration)
            node_name_to_agent_dict = {a.curr_node.xy_name: a for a in self.agents}
            node_name_to_agent_list = list(node_name_to_agent_dict.keys())
            heapq.heapify(node_name_to_agent_list)
            solved = pibt_func(agent=agent, h_dict=self.h_dict, nodes_dict=self.nodes_dict,
                               vc_set=vc_set, ec_set=ec_set,
                               node_name_to_agent_dict=node_name_to_agent_dict,
                               node_name_to_agent_list=node_name_to_agent_list,
                               next_iteration=self.next_iteration,
                               logs=self.logs)

        # others stay on place
        for agent in self.agents:
            if len(agent.path) - 1 != self.next_iteration:
                agent.path.append(agent.path[-1])

        runtime = time.time() - start_time
        self.logs['runtime'] += runtime


@use_profiler(save_dir='../stats/alg_PIBT.pstat')
def main():
    # SACG
    # N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_SACG()
    # LMAPF
    # N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = params_for_LMAPF()
    N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save = ...

    # problem creation
    env = SimEnvLMAPF(img_dir=img_dir, is_sacg=is_sacg)
    start_nodes = random.sample(env.nodes, N)
    plot_rate = 0.5

    # for rendering
    total_unique_moves_list = []
    total_finished_goals_list = []
    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # the run
    obs = env.reset(start_node_names=[n.xy_name for n in start_nodes], max_time=max_time, corridor_size=corridor_size)
    # alg creation + init
    alg = AlgPIBT(env=env, to_check_paths=to_check_paths)
    alg.initiate_problem(obs=obs)

    for i_step in range(max_time):
        actions = alg.get_actions(obs)  # alg part
        obs, metrics, terminated, info = env.step(actions)

        # update metrics + render
        print(f'\ntotal_finished_goals -> {metrics['total_finished_goals']} ')
        total_unique_moves_list.append(metrics['total_unique_moves'])
        total_finished_goals_list.append(metrics['total_finished_goals'])
        if to_render:
            i_agent = alg.global_order[0]
            plot_info = {'i': i_step, 'iterations': max_time, 'img_dir': img_dir, 'img_np': alg.img_np,
                         'n_agents': env.n_agents, 'agents': alg.agents,
                         'total_unique_moves_list': total_unique_moves_list,
                         'total_finished_goals_list': total_finished_goals_list, 'i_agent': i_agent,
                         'corridor': i_agent.path[i_step:]}
            plot_step_in_env(ax[0], plot_info)
            plot_total_finished_goals(ax[1], plot_info)
            # plot_unique_movements(ax[1], plot_info)
            plt.pause(plot_rate)

        if terminated:
            break

    # if to_render:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_info = {'i': max_time, 'iterations': max_time, 'img_dir': img_dir, 'img_np': env.img_np,
                 'n_agents': env.n_agents, 'agents': env.agents, 'total_unique_moves_list': total_unique_moves_list,
                 'total_finished_goals_list': total_finished_goals_list, }
    plot_step_in_env(ax[0], plot_info)
    plot_total_finished_goals(ax[1], plot_info)
    # plot_unique_movements(ax[1], plot_info)
    plt.show()
    if env.is_sacg:
        max_time = len(env.agents_dict['agent_0'].path)
    do_the_animation(info={'img_dir': img_dir, 'img_np': env.img_np, 'agents': alg.agents, 'max_time': max_time,
                           'is_sacg': env.is_sacg, 'alg_name': alg.name}, to_save=to_save)
    print(f'finished run')


if __name__ == '__main__':
    main()
