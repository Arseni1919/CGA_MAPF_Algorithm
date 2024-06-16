import matplotlib.pyplot as plt

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from create_animation import do_the_animation
from environments.env_MAPF import SimEnvMAPF
from algs.alg_b_PIBT import AlgPIBT
from algs.alg_CGAR3_MAPF import AlgCgar3Mapf


def run_mapf_experiments():
    # ------------------------------------------------------------------------------------------------------------ #
    # General params
    # ------------------------------------------------------------------------------------------------------------ #
    # set_seed(random_seed_bool=False, seed=381)
    # set_seed(random_seed_bool=False, seed=9256)  # 500 - room
    set_seed(random_seed_bool=False, seed=1112)
    # set_seed(random_seed_bool=True)

    # ------------------------------------------------------------------------------------------------------------ #
    # MAPF
    # ------------------------------------------------------------------------------------------------------------ #
    n_agents_list = [100, 200, 300, 400]

    i_problems = 5

    alg_list = [AlgPIBT, AlgCgar3Mapf]

    # img_dir = '10_10_my_rand.map'
    # img_dir = '15-15-two-rooms.map'
    # img_dir = '15-15-four-rooms.map'
    # img_dir = '15-15-six-rooms.map'
    # img_dir = '15-15-eight-rooms.map'

    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    img_dir = 'room-32-32-4.map'
    # limits
    # max_time = 1e7  # seconds
    max_time = 60  # seconds
    max_time = 10  # seconds
    # debug
    # to_assert = True
    to_assert = False
    # rendering
    # to_render = True
    to_render = False

    logs_dict = {
        alg.name: {
            f'{n_agents}': {
                'soc': [],
                'makespan': [],
                'sr': [],
                'time': [],
            } for n_agents in n_agents_list
        } for alg in alg_list
    }
    logs_dict['alg_names'] = [alg.name for alg in alg_list]
    logs_dict['n_agents_list'] = n_agents_list
    logs_dict['i_problems'] = i_problems
    logs_dict['img_dir'] = img_dir
    logs_dict['max_time'] = max_time

    # ------------------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------------------ #

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    for n_agents in n_agents_list:

        for i_problem in range(i_problems):

            env = SimEnvMAPF(
                img_dir=img_dir, is_SACGR=False,
                path_to_maps='maps',
                path_to_heuristics='logs_for_heuristics',
                path_to_freedom_maps='logs_for_freedom_maps'
            )
            start_nodes = random.sample(env.nodes, n_agents)
            obs = env.reset(start_node_names=[n.xy_name for n in start_nodes])

            for alg in alg_list:

                # the run
                # alg creation + init
                alg = alg(env=env)
                alg.initialize_problem(obs=obs)
                # solvable, message = alg.check_solvability()
                solved, paths_dict = False, {}
                solved, paths_dict = alg.solve(max_time=max_time, to_assert=to_assert, to_render=to_render)

                if solved:
                    logs_dict[alg.name][f'{n_agents}']['sr'].append(1)
                else:
                    logs_dict[alg.name][f'{n_agents}']['sr'].append(0)
                print(f'\n{n_agents=}, {i_problem=}, {alg.name=}')

            # plot
            plot_sr(ax[0, 0], info=logs_dict)
            plt.pause(0.001)

    print('\n[INFO]: finished BIG experiments')
    plt.show()

# if to_render:
#     fig, ax = plt.subplots(1, 2, figsize=(14, 7))


if __name__ == '__main__':
    run_mapf_experiments()



