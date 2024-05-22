from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from create_animation import do_the_animation
from environments.env_MAPF import SimEnvMAPF


def single_mapf_run(AlgClass, is_SACGR=True):
    # ------------------------------------------------------------------------------------------------------------ #
    # General params
    # ------------------------------------------------------------------------------------------------------------ #
    # set_seed(random_seed_bool=False, seed=381)
    set_seed(random_seed_bool=False, seed=9256)  # 500
    # set_seed(random_seed_bool=False, seed=1111)
    # set_seed(random_seed_bool=True)

    if is_SACGR:
        # ------------------------------------------------------------------------------------------------------------ #
        # SACGR
        # ------------------------------------------------------------------------------------------------------------ #
        N = 500
        i_problems = 10
        # img_dir = '15-15-two-rooms.map'
        # img_dir = '15-15-four-rooms.map'
        # img_dir = '15-15-six-rooms.map'
        # img_dir = '15-15-eight-rooms.map'
        # img_dir = '10_10_my_rand.map'

        # img_dir = 'empty-32-32.map'
        # img_dir = 'random-32-32-10.map'
        # img_dir = 'random-32-32-20.map'
        # img_dir = 'maze-32-32-4.map'
        # img_dir = 'maze-32-32-2.map'
        img_dir = 'room-32-32-4.map'
        # limits
        max_time = 1e7  # seconds
        # debug
        to_check_paths = True
        # to_check_paths = False
        to_assert = True
        # to_assert = False
        # rendering
        to_render = True
        # to_render = False
        to_save_animation = False
        # to_do_animation = True
        to_do_animation = False
    else:
        # ------------------------------------------------------------------------------------------------------------ #
        # MAPF
        # ------------------------------------------------------------------------------------------------------------ #
        N = 500
        i_problems = 1
        # img_dir = '10_10_my_rand.map'
        # img_dir = '15-15-two-rooms.map'
        # img_dir = '15-15-four-rooms.map'
        # img_dir = '15-15-six-rooms.map'
        # img_dir = '15-15-eight-rooms.map'

        # img_dir = 'empty-32-32.map'
        # img_dir = 'random-32-32-10.map'
        img_dir = 'random-32-32-20.map'
        # img_dir = 'maze-32-32-4.map'
        # img_dir = 'maze-32-32-2.map'
        # img_dir = 'room-32-32-4.map'
        # limits
        max_time = 1e7  # seconds
        # debug
        to_check_paths = True
        # to_check_paths = False
        # to_assert = True
        to_assert = False
        # rendering
        # to_render = True
        to_render = False
        # to_save_animation = True
        to_save_animation = False
        to_do_animation = True
        # to_do_animation = False
        # ------------------------------------------------------------------------------------------------------------ #
        # ------------------------------------------------------------------------------------------------------------ #
        # ------------------------------------------------------------------------------------------------------------ #

    for i_problem in range(i_problems):
        # if i_problem >= 4:
        #     to_render = True
        # problem creation
        env = SimEnvMAPF(img_dir=img_dir, is_SACGR=is_SACGR)
        start_nodes = random.sample(env.nodes, N)
        obs = env.reset(start_node_names=[n.xy_name for n in start_nodes])

        # the run
        # alg creation + init
        alg = AlgClass(env=env)
        alg.initialize_problem(obs=obs)
        solvable, message = alg.check_solvability()
        print(f'Solvability: {message}')

        solved, paths_dict = False, {}
        if solvable:
            solved, paths_dict = alg.solve(max_time=max_time, to_assert=to_assert, to_render=to_render)

        if to_check_paths:
            # TODO
            pass

        plt.close()
        if solved and to_do_animation:
            do_the_animation(info={
                'img_dir': img_dir, 'img_np': env.img_np, 'paths_dict': paths_dict, 'i_agent': alg.agents[0],
                'max_time': len(max(list(paths_dict.values()))), 'alg_name': alg.name
            }, to_save=to_save_animation)
        print(f'The run {i_problem + 1} is finished\n{solved=}')

# if to_render:
#     fig, ax = plt.subplots(1, 2, figsize=(14, 7))



