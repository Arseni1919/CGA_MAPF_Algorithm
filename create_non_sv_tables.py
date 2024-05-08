from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *


def parallel_get_non_sv_nodes_np[T](node: T, nodes: List[Node], nodes_dict: Dict[str, Node], img_np: np.ndarray, non_sv_nodes_np_dict: Dict[str, np.ndarray]) -> None:
    print(f'\n{node.xy_name} started...', end='')
    blocked_nodes = [node]
    non_sv_nodes_np = get_non_sv_nodes_np(nodes, nodes_dict, img_np, blocked_nodes=blocked_nodes)
    non_sv_nodes_np_dict[node.xy_name] = non_sv_nodes_np
    print(f'\n{node.xy_name} is finished.', end='')


def create_non_sv_nodes_np_dict(nodes: List[Node], nodes_dict: Dict[str, Node], img_np: np.ndarray):
    non_sv_nodes_np_dict: Dict[str, np.ndarray] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        for node in nodes:
            executor.submit(parallel_get_non_sv_nodes_np, node, nodes, nodes_dict, img_np, non_sv_nodes_np_dict)


def main():
    pass


if __name__ == '__main__':
    main()