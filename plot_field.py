from tools_for_plotting import *
from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *


def plot_just_field(ax, info):
    img_np = info['img_np']
    field = img_np
    # field = img_np * -1
    ax.imshow(field, origin='lower', cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def main():
    img_dir = '15-15-two-rooms.map'
    # img_dir = '15-15-four-rooms.map'
    # img_dir = '15-15-six-rooms.map'
    # img_dir = '15-15-eight-rooms.map'

    nodes, nodes_dict, img_np = build_graph_nodes(img_dir=img_dir, path='maps', show_map=False)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_just_field(ax, info={'img_np': img_np})
    plt.show()


if __name__ == '__main__':
    main()






