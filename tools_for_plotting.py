import matplotlib.pyplot as plt
import numpy as np

from globals import *
from functions import get_color


def plot_magnet_field(path, data):
    plt.rcParams["figure.figsize"] = [8.00, 8.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot field
    if data is not None:
        x_l, y_l, z_l = np.nonzero(data > 0)
        col = data[data > 0]
        alpha_col = col / max(col) if len(col) > 0 else 1
        # alpha_col = np.exp(col) / max(np.exp(col))
        cm = plt.colormaps['Reds']  # , cmap=cm
        ax.scatter(x_l, y_l, z_l, c=col, alpha=alpha_col, marker='s', cmap=cm)
    # plot line
    if path:
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        path_z = list(range(len(path_x)))
        ax.plot(path_x, path_y, path_z)
    plt.show()
    # plt.pause(2)


def get_line_marker(index, kind):
    if kind == 'l':
        lines = ['--', '-', '-.', ':']
        index = index % len(lines)
        return lines[index]
    elif kind == 'm':
        markers = ['^', '1', '2', 'X', 'd', 'v', 'o']
        index = index % len(markers)
        return markers[index]
    else:
        raise RuntimeError('no such kind')


def set_plot_title(ax, title, size=9):
    ax.set_title(f'{title}', fontweight="bold", size=size)


def set_log(ax):
    # log = True
    log = False
    if log:
        ax.set_yscale('log')
    return log


def plot_text_in_cactus(ax, l_x, l_y):
    if len(l_x) > 0:
        ax.text(l_x[-1] - 2, l_y[-1], f'{l_x[-1] + 1}', bbox=dict(facecolor='yellow', alpha=0.75))


def set_legend(ax, framealpha=None, size=9):
    to_put_legend = True
    # to_put_legend = False
    if to_put_legend:
        if not framealpha:
            framealpha = 0
        legend_properties = {'weight': 'bold', 'size': size}
        # legend_properties = {}
        if framealpha is not None:
            ax.legend(prop=legend_properties, framealpha=framealpha)
        else:
            ax.legend(prop=legend_properties)


# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #


def plot_unique_movements(ax, info):
    ax.cla()
    total_unique_moves_list = info['total_unique_moves_list']
    ax.plot(total_unique_moves_list)
    # ax.set_xlim([0, iterations])
    ax.set_xlabel('iters')
    ax.set_ylabel('unique moves')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'title', size=10)
    # set_legend(ax, size=12)


def plot_total_finished_goals(ax, info):
    ax.cla()
    total_finished_goals_list = info['total_finished_goals_list']
    iterations = info['iterations']
    ax.plot(total_finished_goals_list)
    ax.set_xlim([0, iterations])
    ax.set_xlabel('iters')
    ax.set_ylabel('Total Finished Goals')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'title', size=10)
    # set_legend(ax, size=12)


def plot_temp_a_star(ax, info):
    ax.cla()
    img_np = info['img_np']
    path = info['path']

    field = img_np * -1
    for n in path:
        field[n.x, n.y] = 3

    if 'open_list' in info:
        open_list = info['open_list']
        for _, n in open_list:
            field[n.x, n.y] = 1

    if 'closed_list' in info:
        closed_list = info['closed_list']
        for _, n in closed_list:
            field[n.x, n.y] = -2

    ax.imshow(field, origin='lower')
    ax.set_title(f'Temporal A*')


def plot_flow_in_env(ax, info):
    ax.cla()
    img_np = info['img_np']
    agents = info['agents']
    corridor = info['corridor']

    field = img_np * -1
    for n in corridor:
        field[n.x, n.y] = 2

    if 'free_nodes' in info:
        free_nodes = info['free_nodes']
        for n in free_nodes:
            field[n.x, n.y] = 3

    # if 'tubes_to_corridor' in info:
    #     tubes_to_corridor = info['tubes_to_corridor']
    #     for tube in tubes_to_corridor:
    #         for n in tube:
    #             field[n.x, n.y] = 5

    if 'tube' in info:
        tube = info['tube']
        for n in tube:
            field[n.x, n.y] = 4
    ax.imshow(field, origin='lower')

    others_y_list, others_x_list, others_cm_list = [], [], []
    for agent in agents:
        curr_node = agent.path[-1]
        others_y_list.append(curr_node.y)
        others_x_list.append(curr_node.x)
        others_cm_list.append(get_color(agent.num))

    ax.scatter(others_y_list, others_x_list, s=100, c='k')
    ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list))
    # ax.scatter(others_y_list, others_x_list, s=50, c='yellow')

    if 'next_agent' in info:
        next_agent = info['next_agent']
        curr_node = next_agent.curr_node
        ax.scatter([curr_node.y], [curr_node.x], s=200, c='w')
        ax.scatter([curr_node.y], [curr_node.x], s=30, c='r')

    ax.set_title(f'title')


def plot_step_in_env(ax, info):
    ax.cla()
    # nodes = info['nodes']
    # a_name = info['i_agent'].name if 'i_agent' in info else 'agent_0'
    img_np = info['img_np']
    agents = info['agents']
    non_sv_nodes_np = info['non_sv_nodes_np'] if 'non_sv_nodes_np' in info else None

    field = img_np * -1
    if non_sv_nodes_np is not None:
        field += 0.5 * non_sv_nodes_np * -1
    if 'corridor' in info:
        corridor = info['corridor']
        for n in corridor:
            field[n.x, n.y] = 2
    if 'occupied_nodes' in info:
        occupied_nodes = info['occupied_nodes']
        for n in occupied_nodes:
            field[n.x, n.y] = 0
    ax.imshow(field, origin='lower', cmap='binary')

    others_y_list, others_x_list, others_cm_list, alpha_list = [], [], [], []
    for agent in agents:
        if 'i_agent' in info and info['i_agent'] == agent:
            continue
        curr_node = agent.curr_node
        others_y_list.append(curr_node.y)
        others_x_list.append(curr_node.x)
        others_cm_list.append(get_color(agent.num))
        if agent.curr_node == agent.goal_node:
            alpha_list.append(0.2)
        else:
            alpha_list.append(1)
    ax.scatter(others_y_list, others_x_list, s=100, c='k', alpha=alpha_list)
    ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list), alpha=alpha_list)
    # ax.scatter(others_y_list, others_x_list, s=50, c='yellow')

    if 'i_agent' in info:
        i_agent = info['i_agent']
        curr_node = i_agent.curr_node
        next_goal_node = i_agent.get_goal_node()
        ax.scatter([curr_node.y], [curr_node.x], s=200, c='k')
        ax.scatter([curr_node.y], [curr_node.x], s=100, c='r')
        ax.scatter([next_goal_node.y], [next_goal_node.x], s=400, c='white', marker='X', alpha=0.4)
        ax.scatter([next_goal_node.y], [next_goal_node.x], s=200, c='red', marker='X', alpha=0.4)

    title_str = 'plot_step_in_env\n'
    if 'to_title' in info:
        to_title = info['to_title']
        title_str += f'{to_title}\n '
    if 'img_dir' in info:
        img_dir = info['img_dir']
        title_str += f'Map: {img_dir[:-4]}\n '
    if 'n_agents' in info:
        n_agents = info['n_agents']
        title_str += f'{n_agents} agents '
    if 'i' in info:
        i = info['i']
        title_str += f'(iteration: {i + 1})'
    ax.set_title(title_str)


def plot_return_paths(ax, info):
    ax.cla()
    # nodes = info['nodes']
    # a_name = info['i_agent'].name if 'i_agent' in info else 'agent_0'
    img_np = info['img_np']
    agents = info['agents']

    field = img_np * -1
    if 'corridor' in info:
        corridor = info['corridor']
        for n in corridor:
            field[n.x, n.y] = 2
    if 'occupied_nodes' in info:
        occupied_nodes = info['occupied_nodes']
        for n in occupied_nodes:
            field[n.x, n.y] = 0
    if 'agents_to_return_dict' in info:
        agents_to_return_dict = info['agents_to_return_dict']
        i_agent = info['i_agent']
        agents_to_plot = agents_to_return_dict[i_agent.name]
    else:
        agents_to_plot = agents

    ax.imshow(field, origin='lower', cmap='binary')

    others_y_list, others_x_list, others_cm_list = [], [], []
    for agent in agents_to_plot:
        if 'i_agent' in info and info['i_agent'] == agent:
            continue
        rn_x_list, rn_y_list = [], []
        # return_path_nodes = agent.return_path_nodes
        return_path = agent.return_road_nodes
        for n in return_path:
            rn_x_list.append(n.x)
            rn_y_list.append(n.y)
        color = get_color(agent.num)
        ax.plot(rn_y_list, rn_x_list, '*-', c=color)
    # ax.scatter(others_y_list, others_x_list, s=100, c='k')
    # ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list))
    # ax.scatter(others_y_list, others_x_list, s=50, c='yellow')

    if 'i_agent' in info:
        i_agent = info['i_agent']
        curr_node = i_agent.curr_node
        next_goal_node = i_agent.get_goal_node()
        ax.scatter([curr_node.y], [curr_node.x], s=120, c='w')
        ax.scatter([curr_node.y], [curr_node.x], s=70, c='r')
        ax.scatter([next_goal_node.y], [next_goal_node.x], s=400, c='white', marker='X', alpha=0.4)
        ax.scatter([next_goal_node.y], [next_goal_node.x], s=200, c='red', marker='X', alpha=0.4)

    title_str = 'plot of return_nodes\n'
    ax.set_title(title_str)


def plot_env_field(ax, info):
    ax.cla()
    # nodes = info['nodes']
    # a_name = info['i_agent'].name if 'i_agent' in info else 'agent_0'
    iterations = info["iterations"]
    n_agents = info['n_agents']
    img_dir = info['img_dir']
    map_dim = info['map_dim']
    img_np = info['img_np']
    curr_iteration = info["i"]
    i_problem = info['i_problem']
    n_problems = info['n_problems']
    agents_names = info['agents_names']
    agents_names.sort()
    sds_plot = 'orders_dict' in info
    if sds_plot:
        orders_dict = info['orders_dict']
        one_master = info['one_master']
    else:
        one_master = info['i_agent']
    # one_master = 'agent_0'

    field = img_np * -1
    others_y_list, others_x_list, others_cm_list = [], [], []
    a_y_list, a_x_list, a_cm_list = [], [], []
    g_y_list, g_x_list, g_cm_list = [], [], []
    for i, agent_name in enumerate(agents_names):
        curr_node = info[agent_name]['curr_node']
        if agent_name == one_master.name:
        # if agent_name == one_master:
            a_x_list.append(curr_node.x)
            a_y_list.append(curr_node.y)
            if sds_plot:
                a_cm_list.append(get_color(orders_dict[agent_name]))
            else:
                a_cm_list.append('k')
            next_goal_node = info[agent_name]['next_goal_node']
            g_x_list.append(next_goal_node.x)
            g_y_list.append(next_goal_node.y)
        else:
            others_y_list.append(curr_node.y)
            others_x_list.append(curr_node.x)
            if sds_plot:
                others_cm_list.append(get_color(orders_dict[agent_name]))
            else:
                others_cm_list.append(get_color(agents_names.index(agent_name)))
    ax.scatter(a_y_list, a_x_list, s=200, c='white')
    ax.scatter(a_y_list, a_x_list, s=100, c=np.array(a_cm_list))
    ax.scatter(g_y_list, g_x_list, s=200, c='white', marker='X')
    ax.scatter(g_y_list, g_x_list, s=100, c='red', marker='X')
    ax.scatter(others_y_list, others_x_list, s=100, c='k')
    ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list))
    # ax.scatter(others_y_list, others_x_list, s=50, c='yellow')

    ax.imshow(field, origin='lower')
    ax.set_title(f'Map: {img_dir[:-4]}\n '
                 f'{n_agents} agents'  # , selected: {one_master.name} - {one_master.order}\n
                 f'(run:{i_problem + 1}/{n_problems}, time: {curr_iteration + 1}/{iterations})')


def plot_magnet_agent_view(ax, info):
    ax.cla()
    # paths_dict = info['paths_dict']
    agent = info['i_agent']
    nodes = info['i_nodes']
    side_x, side_y = info['map_dim']
    t = info['i']

    field = np.zeros((side_x, side_y))

    # magnet field
    if agent.nei_pfs is not None:
        if nodes:
            for node in nodes:
                field[node.x, node.y] = agent.nei_pfs[node.x, node.y, 0]

    # an agent
    ax.scatter(agent.curr_node.y, agent.curr_node.x, s=200, c='white')
    ax.scatter(agent.curr_node.y, agent.curr_node.x, s=100, c='k')
    ax.scatter(agent.next_goal_node.y, agent.next_goal_node.x, s=200, c='white', marker='X')
    ax.scatter(agent.next_goal_node.y, agent.next_goal_node.x, s=100, c='red', marker='X')

    # agent's nei poses
    # x_path = [node.x for node in agent.nei_nodes]
    # y_path = [node.y for node in agent.nei_nodes]
    # ax.scatter(y_path, x_path, c='green', alpha=0.05)


    # its path
    if agent.plan is not None and len(agent.plan) > 0:
        x_path = [node.x for node in agent.plan]
        y_path = [node.y for node in agent.plan]
        # ax.plot(x_path, y_path, c='yellow')
        ax.plot(y_path, x_path, c='blue')

    ax.imshow(field, origin='lower', cmap='hot')
    ax.set_title(f"{agent.name}'s View (time: {t})")


def plot_step_in_mapf_paths(ax, info):
    ax.cla()
    paths_dict = info['paths_dict']
    nodes = info['nodes']
    side_x = info['side_x']
    side_y = info['side_y']
    t = info['t']
    img_dir = info['img_dir']
    a_name = info['agent'].name if 'agent' in info else 'agent_0'
    longest_path = info['longest_path']

    field = np.zeros((side_x, side_y))

    if nodes:
        for node in nodes:
            field[node.x, node.y] = -1

    n = len(list(paths_dict.keys()))
    color_map = plt.cm.get_cmap('hsv', n)
    i = 0
    for agent_name, path in paths_dict.items():
        t_path = path[:t + 1]
        # for node in t_path:
        #     field[node.x, node.y] = 3
        if agent_name == a_name:
            ax.scatter(t_path[-1].y, t_path[-1].x, s=200, c='white')
            ax.scatter(t_path[-1].y, t_path[-1].x, s=100, c='k')
        else:
            ax.scatter(t_path[-1].y, t_path[-1].x, s=100, c='k')
            ax.scatter(t_path[-1].y, t_path[-1].x, s=50, c=np.array([color_map(i)]))
        # ax.text(t_path[-1].y - 0.4, t_path[-1].x - 0.4, agent_name[6:])
        i += 1

    # for agent_name, path in paths_dict.items():
    #     # field[path[0].x, path[0].y] = 4
    #     field[path[-1].x, path[-1].y] = 5

    ax.imshow(field, origin='lower')
    ax.set_title(f'Map: {img_dir[:-4]}, N_agents: {n} (time: {t}/{longest_path})')


def plot_sr(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg in alg_names:
        sr_list = []
        x_list = []
        for n_a in n_agents_list:
            if len(info[i_alg][f'{n_a}']['sr']) > 0:
                sr_list.append(np.sum(info[i_alg][f'{n_a}']['sr']) / len(info[i_alg][f'{n_a}']['sr']))
                x_list.append(n_a)
        ax.plot(x_list, sr_list, markers_lines_dict[i_alg], color=colors_dict[i_alg],
                alpha=0.5, label=f'{i_alg}', linewidth=5, markersize=20)
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_ylim([0, 1 + 0.1])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', fontsize=15)
    ax.set_ylabel('Success Rate', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    set_plot_title(ax, f'{img_dir[:-4]} Map', size=11)
    set_legend(ax, size=17)
    plt.tight_layout()


def plot_soc(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg in alg_names:
        soc_list = []
        for n_a in n_agents_list:
            soc_list.append(np.mean(info[i_alg][f'{n_a}']['soc']))
        ax.plot(n_agents_list, soc_list, markers_lines_dict[i_alg], color=colors_dict[i_alg], alpha=0.5, label=f'{i_alg}')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average SoC')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=10)
    set_legend(ax, size=12)


def plot_rsoc(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg_no_pf, i_alg in [('PrP', 'PF-PrP'), ('LNS2', 'PF-LNS2')]:
        no_pf_soc_list = []
        for n_a in n_agents_list:
            no_pf_soc_list.append(np.mean(info[i_alg_no_pf][f'{n_a}']['soc']))
        no_pf_soc_list = np.array(no_pf_soc_list)
        soc_list = []
        for n_a in n_agents_list:
            soc_list.append(np.mean(info[i_alg][f'{n_a}']['soc']))
        soc_list = np.array(soc_list)
        y_list = soc_list / no_pf_soc_list
        ax.plot(n_agents_list, y_list, markers_lines_dict[i_alg], color=colors_dict[i_alg], alpha=0.5, label=f'{i_alg}')

        # print
        print(f'{i_alg}')
        for n_a, y_val in zip(n_agents_list, y_list):
            print(f'{n_a} -> {y_val: .2f}')

    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average RSoC')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=10)
    set_legend(ax, size=12)


def plot_time_metric(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    # x_list = n_agents_list[:4]
    x_list = n_agents_list
    for i_alg in alg_names:
        soc_list = []
        res_str = ''
        for n_a in x_list:
            soc_list.append(np.mean(info[i_alg][f'{n_a}']['time']))
            res_str += f'\t{n_a} - {soc_list[-1]: .2f}, '
        ax.plot(x_list, soc_list, markers_lines_dict[i_alg], color=colors_dict[i_alg],
                alpha=0.5, label=f'{i_alg}', linewidth=5, markersize=20)
        print(f'{i_alg}\t\t\t: {res_str}')
    ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    ax.set_xticks(x_list)
    ax.set_xlabel('N agents', fontsize=15)
    ax.set_ylabel('Average Runtime', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=11)
    set_legend(ax, size=17)


def plot_time_metric_cactus(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    # x_list = n_agents_list[:4]
    x_list = n_agents_list
    for i_alg in alg_names:
        rt_list = []
        # res_str = ''
        for n_a in x_list:
            rt_list.extend(info[i_alg][f'{n_a}']['runtime'])
            # res_str += f'\t{n_a} - {rt_list[-1]: .2f}, '
        rt_list.sort()
        ax.plot(rt_list, markers_lines_dict[i_alg], color=colors_dict[i_alg],
                alpha=0.5, label=f'{i_alg}', linewidth=2, markersize=10)
        # print(f'{i_alg}\t\t\t: {res_str}')
    # ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    # ax.set_xticks(x_list)
    ax.set_xlabel('Solved Instances', fontsize=15)
    ax.set_ylabel('Runtime', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    set_plot_title(ax, f'{img_dir[:-4]} Map', size=11)
    set_legend(ax, size=17)
    plt.tight_layout()


def plot_en_metric_cactus(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    # x_list = n_agents_list[:4]
    x_list = n_agents_list
    for i_alg in alg_names:
        rt_list = []
        # res_str = ''
        for n_a in x_list:
            rt_list.extend(info[i_alg][f'{n_a}']['expanded_nodes'])
            # res_str += f'\t{n_a} - {rt_list[-1]: .2f}, '
        rt_list.sort()
        ax.plot(rt_list, markers_lines_dict[i_alg], color=colors_dict[i_alg],
                alpha=0.5, label=f'{i_alg}', linewidth=2, markersize=10)
        # print(f'{i_alg}\t\t\t: {res_str}')
    # ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    # ax.set_xticks(x_list)
    ax.set_xlabel('Solved Instances', fontsize=15)
    ax.set_ylabel('Expanded Nodes', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    set_plot_title(ax, f'{img_dir[:-4]} Map', size=11)
    set_legend(ax, size=17)
    plt.tight_layout()


def plot_sq_metric_cactus(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    # x_list = n_agents_list[:4]
    x_list = n_agents_list
    for i_alg in alg_names:
        rt_list = []
        # res_str = ''
        for n_a in x_list:
            rt_list.extend(info[i_alg][f'{n_a}']['sq'])
            # res_str += f'\t{n_a} - {rt_list[-1]: .2f}, '
        rt_list.sort()
        ax.plot(rt_list, markers_lines_dict[i_alg], color=colors_dict[i_alg],
                alpha=0.5, label=f'{i_alg}', linewidth=2, markersize=10)
        # print(f'{i_alg}\t\t\t: {res_str}')
    # ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    # ax.set_xticks(x_list)
    ax.set_xlabel('Solved Instances', fontsize=15)
    ax.set_ylabel('SoC', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    set_plot_title(ax, f'{img_dir[:-4]} Map', size=11)
    set_legend(ax, size=17)
    plt.tight_layout()


def plot_makespan(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg in alg_names:
        makespan_list = []
        for n_a in n_agents_list:
            makespan_list.append(np.mean(info[i_alg][f'{n_a}']['makespan']))
        ax.plot(n_agents_list, makespan_list, '-^', label=f'{i_alg}')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average Makespan')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=10)
    set_legend(ax, size=12)


def plot_throughput(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']
    max_time = info['max_time']

    for i_alg in alg_names:
        y_list = []
        for n_a in n_agents_list:
            y_list.append(np.mean(info[i_alg][f'{n_a}']['throughput']))
        print(f'{i_alg} -> {y_list}')
        ax.plot(n_agents_list, y_list, markers_lines_dict[i_alg], color=colors_dict[i_alg],
                alpha=0.5, label=f'{i_alg}', linewidth=3, markersize=13)
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', fontsize=15)
    ax.set_ylabel('Average Throughput', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {iterations} iters.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {max_time} steps.',
                   size=11)
    set_legend(ax, size=11)


def plot_max_waiting(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']
    max_time = info['max_time']

    for i_alg in alg_names:
        y_list = []
        for n_a in n_agents_list:
            y_list.append(np.mean(info[i_alg][f'{n_a}']['max_waiting']))
        ax.plot(n_agents_list, y_list, markers_lines_dict[i_alg], color=colors_dict[i_alg],
                alpha=0.5, label=f'{i_alg}', linewidth=3, markersize=13)
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', fontsize=15)
    ax.set_ylabel('Maximum Waiting', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {iterations} iters.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {max_time} steps.',
                   size=11)
    set_legend(ax, size=11)


def plot_lmapf_time(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']
    iterations = info['iterations']

    for i_alg in alg_names:
        y_list = []
        for n_a in n_agents_list:
            y_list.append(np.mean(info[i_alg][f'{n_a}']['time']))
        ax.plot(n_agents_list, y_list, markers_lines_dict[i_alg], color=colors_dict[i_alg], label=i_alg)
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average Runtime')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {iterations} iters.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {iterations} iters.',
                   size=10)
    set_legend(ax, size=12)
    # ax.set_xlabel('N agents', labelpad=-1)
    # ax.set_ylabel('Average Throughput', labelpad=-1)
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
class Plotter:
    def __init__(self, map_dim=None, subplot_rows=2, subplot_cols=4, online_plotting=True):
        if map_dim:
            self.side_x, self.side_y = map_dim
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        if online_plotting:
            self.fig, self.ax = plt.subplots(subplot_rows, subplot_cols, figsize=(14, 7))

    def close(self):
        plt.close()

    # online
    def plot_magnets_run(self, **kwargs):
        info = {
            'agent': kwargs['agent'],
            'paths_dict': kwargs['paths_dict'], 'nodes': kwargs['nodes'],
            'side_x': self.side_x, 'side_y': self.side_y, 't': kwargs['t'],
            'img_dir': kwargs['img_dir'] if 'img_dir' in kwargs else '',
        }
        plot_step_in_mapf_paths(self.ax[0], info)
        plot_magnet_agent_view(self.ax[1], info)
        plt.pause(0.001)
        # plt.pause(1)

    def plot_lists(self, open_list, closed_list, start, goal=None, path=None, nodes=None, a_star_run=False, **kwargs):
        plt.close()
        self.fig, self.ax = plt.subplots(1, 3, figsize=(14, 7))
        field = np.zeros((self.side_x, self.side_y))

        if nodes:
            for node in nodes:
                field[node.x, node.y] = -1

        for node in open_list:
            field[node.x, node.y] = 1

        for node in closed_list:
            field[node.x, node.y] = 2

        if path:
            for node in path:
                field[node.x, node.y] = 3

        field[start.x, start.y] = 4
        if goal:
            field[goal.x, goal.y] = 5

        self.ax[0].imshow(field, origin='lower')
        self.ax[0].set_title('general')

        # if path:
        #     for node in path:
        #         field[node.x, node.y] = 3
        #         self.ax[0].text(node.x, node.y, f'{node.ID}', bbox={'facecolor': 'yellow', 'alpha': 1, 'pad': 10})

        # open_list
        field = np.zeros((self.side_x, self.side_y))
        for node in open_list:
            if a_star_run:
                field[node.x, node.y] = node.g
            else:
                field[node.x, node.y] = node.g_dict[start.xy_name]
        self.ax[1].imshow(field, origin='lower')
        self.ax[1].set_title('open_list')

        # closed_list
        field = np.zeros((self.side_x, self.side_y))
        for node in closed_list:
            if a_star_run:
                field[node.x, node.y] = node.g
            else:
                field[node.x, node.y] = node.g_dict[start.xy_name]
        self.ax[2].imshow(field, origin='lower')
        self.ax[2].set_title('closed_list')

        self.fig.tight_layout()
        # plt.pause(1)
        # plt.pause(0.01)
        self.fig.suptitle(f'{kwargs["agent_name"]}', fontsize=16)
        # self.fig.suptitle(f'suptitle', fontsize=16)
        plt.show()

    def plot_mapf_paths(self, paths_dict, nodes=None, **kwargs):
        plt.close()
        plt.rcParams["figure.figsize"] = [7.00, 7.00]
        # plt.rcParams["figure.autolayout"] = True
        plot_per = kwargs['plot_per']
        plot_rate = kwargs['plot_rate']
        self.fig, self.ax = plt.subplots()
        longest_path = max([len(path) for path in paths_dict.values()])

        for t in range(longest_path):
            if t % plot_per == 0:
                info = {
                    'paths_dict': paths_dict, 'nodes': nodes,
                    'side_x': self.side_x, 'side_y': self.side_y, 't': t,
                    'img_dir': kwargs['img_dir'] if 'img_dir' in kwargs else '',
                    'longest_path': longest_path,
                }
                plot_step_in_mapf_paths(self.ax, info)
                # plt.pause(1)
                plt.pause(plot_rate)