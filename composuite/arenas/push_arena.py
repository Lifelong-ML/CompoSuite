from composuite.utils.mjk_utils import xml_path_completion
from composuite.arenas.compositional_arena import CompositionalArena


class PushArena(CompositionalArena):
    """Workspace for the Push task. 
    """

    def __init__(
        self, bin1_pos=None, bin2_pos=None
    ):
        super().__init__(xml_path_completion("arenas/push_arena.xml"), bin1_pos=bin1_pos,
                         bin2_pos=bin2_pos)

        
