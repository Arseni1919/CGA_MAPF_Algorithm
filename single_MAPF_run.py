from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from create_animation import do_the_animation
from environments.env_MAPF import SimEnvMAPF


def single_mapf_run(AlgClass, is_SACGR=True):
    # ------------------------------------------------------------------------------------------------------------ #
    # General params
    # ------------------------------------------------------------------------------------------------------------ #
    set_seed(random_seed_bool=False, seed=9922)

    if is_SACGR:
        # ------------------------------------------------------------------------------------------------------------ #
        # SACGR
        # ------------------------------------------------------------------------------------------------------------ #
        N = 100
        img_dir = 'empty-32-32.map'
        # img_dir = 'random-32-32-20.map'
        # limits
        max_time = 1e7  # seconds
        # debug
        to_check_paths = True
        # to_check_paths = False
        to_assert = True
        # rendering
        to_render = True
        # to_render = False
        to_save_animation = False
    else:
        # ------------------------------------------------------------------------------------------------------------ #
        # MAPF
        # ------------------------------------------------------------------------------------------------------------ #
        N = 100
        # img_dir = 'empty-32-32.map'
        img_dir = 'random-32-32-20.map'
        # limits
        max_time = 1e7  # seconds
        # debug
        to_check_paths = True
        to_check_paths = False
        to_assert = True
        # rendering
        to_render = True
        # to_render = False
        to_save_animation = False

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

    solved = False
    if solvable:
        solved, paths_dict = alg.solve(max_time=max_time, to_assert=to_assert, to_render=to_render)

    if to_check_paths:
        pass

    if solved:
        do_the_animation(info={'img_dir': img_dir, 'img_np': env.img_np, 'agents': alg.agents, 'max_time': max_time, 'alg_name': alg.name}, to_save=to_save_animation)
    print(f'The run is finished\n{solved=}')

# if to_render:
#     fig, ax = plt.subplots(1, 2, figsize=(14, 7))



