import numpy as np

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *

# load
# if img_dir is not None:
#     possible_dir = f'{folder_dir}/{img_dir[:-4]}.npy'
#     if os.path.exists(possible_dir):
#         with open(possible_dir, 'rb') as f:
#             non_sv_nodes_np = np.load(f)
#             return non_sv_nodes_np

# save
# if img_dir is not None:
#     if os.path.exists('logs_for_freedom_maps'):
#         with open(possible_dir, 'wb') as f:
#             np.save(f, non_sv_nodes_np)
#         # print('Saved freedom nodes.')


def get_blocked_non_sv_nodes(img_dir: str, folder_dir: str = 'logs_for_freedom_maps'):
    possible_dir = f'{folder_dir}/blocked_{img_dir[:-4]}.npy'
    assert os.path.exists(possible_dir)
    with open(possible_dir, 'rb') as f:
        non_sv_nodes_with_blocked_np = np.load(f)
        return non_sv_nodes_with_blocked_np


def get_non_sv_nodes_with_blocked_np[T](node: T, nodes: List[Node], nodes_dict: Dict[str, Node], img_np: np.ndarray, non_sv_nodes_with_blocked_np: np.ndarray) -> None:
    print(f'{node.xy_name} started...', end='')
    blocked_nodes = [node]
    non_sv_nodes_np = get_non_sv_nodes_np(nodes, nodes_dict, img_np, blocked_nodes=blocked_nodes, img_dir=None)
    non_sv_nodes_with_blocked_np[node.x, node.y, :, :] = non_sv_nodes_np
    print(f'   finished.')


def create_non_sv_nodes_with_blocked_np(nodes: List[Node], nodes_dict: Dict[str, Node], img_np: np.ndarray, img_dir: str):
    # x, y, x, y
    print(f'Started to create blocked_{img_dir[:-4]}.npy...')
    non_sv_nodes_with_blocked_np: np.ndarray = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[0], img_np.shape[1]))
    for node in nodes:
        get_non_sv_nodes_with_blocked_np(node, nodes, nodes_dict, img_np, non_sv_nodes_with_blocked_np)
    assert os.path.exists('logs_for_freedom_maps')
    possible_dir = f'logs_for_freedom_maps/blocked_{img_dir[:-4]}.npy'
    with open(possible_dir, 'wb') as f:
        np.save(f, non_sv_nodes_with_blocked_np)
        print(f'Saved freedom nodes of {img_dir} (with blocked options) to {possible_dir}.')



def main():
    img_dir = '10_10_my_rand.map'
    # img_dir = '15-15-two-rooms.map'
    # img_dir = '15-15-four-rooms.map'
    # img_dir = '15-15-six-rooms.map'
    # img_dir = '15-15-eight-rooms.map'

    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'room-32-32-4.map'

    path_to_maps: str = 'maps'
    map_dim = get_dims_from_pic(img_dir=img_dir, path=path_to_maps)
    nodes, nodes_dict, img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
    # non_sv_nodes_np: np.ndarray = get_non_sv_nodes_np(nodes, nodes_dict, img_np, img_dir=img_dir, folder_dir='../logs_for_freedom_maps')
    create_non_sv_nodes_with_blocked_np(nodes, nodes_dict, img_np, img_dir=img_dir)

    non_sv_nodes_with_blocked_np = get_blocked_non_sv_nodes(img_dir=img_dir)
    print()


if __name__ == '__main__':
    main()

