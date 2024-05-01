import abc
from abc import ABC

from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF


class AlgGeneric(ABC):
    name = 'Generic'

    def __init__(self):
        pass

    @abc.abstractmethod
    def initialize_problem(self, obs: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def check_solvability(self) -> Tuple[bool, str]:
        pass

    @abc.abstractmethod
    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[bool, Dict[str, List[Node]]]:
        pass


