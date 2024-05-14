from tools_for_plotting import *
from tools_for_heuristics import *
from tools_for_graph_nodes import *
from single_MAPF_run import single_mapf_run
from environments.env_MAPF import SimEnvMAPF
from algs.alg_generic_class import AlgGeneric
from algs.alg_PIBT import run_i_pibt


class AlgCgaMapf(AlgGeneric):

    name = 'CGA-MAPF'

    def initialize_problem(self, obs: Dict[str, Any]) -> None:
        pass

    def check_solvability(self) -> Tuple[bool, str]:
        pass

    def solve(self, max_time: int, to_assert: bool = True, to_render: bool = False) -> Tuple[
        bool, Dict[str, List[Node]]]:
        pass

    def stop_condition(self):
        for agent in self.agents:
            if agent.path[-1] != agent.goal_node:
                return False
        return True

