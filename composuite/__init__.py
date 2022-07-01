import os 
import robosuite
from composuite.tasks.pick_place_subtask import PickPlaceSubtask
from composuite.tasks.push_subtask import PushSubtask
from composuite.tasks.shelf_subtask import ShelfSubtask
from composuite.tasks.trashcan_subtask import TrashcanSubtask

robosuite.environments.base.register_env(PickPlaceSubtask)
robosuite.environments.base.register_env(PushSubtask)
robosuite.environments.base.register_env(ShelfSubtask)
robosuite.environments.base.register_env(TrashcanSubtask)

from composuite.env.main import make, sample_tasks
assets_root = os.path.join(os.path.dirname(__file__), "assets")
