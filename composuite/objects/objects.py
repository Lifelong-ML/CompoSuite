from robosuite.models.objects import MujocoXMLObject
from composuite.utils.mjk_utils import xml_path_completion

import numpy as np


class ObjectWallObject(MujocoXMLObject):
    """
    Wall obstacle (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/object_wall.xml"),
                         name=name, obj_type="all", joints=None, duplicate_collision_geoms=True)

        self.wall_size = np.array(list(map(float, self.worldbody.findall(
            "./body/body/geom")[0].items()[3][1].split())))


class ObjectDoorFrameObject(MujocoXMLObject):
    """
    DoorFrame obstacle (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/object_door_frame.xml"),
                         name=name, obj_type="all", joints=None, duplicate_collision_geoms=True)

        self.door_l_size = np.array(list(map(float, self.worldbody.findall(
            "./body/body/body/geom")[0].items()[3][1].split())))


class GoalWallObject(MujocoXMLObject):
    """
    Wall obstacle (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/goal_wall.xml"),
                         name=name, obj_type="all", joints=None, duplicate_collision_geoms=True)


class DumbbellObject(MujocoXMLObject):
    """
    Cup object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/dumbbell.xml"),
                         name=name, obj_type="all", joints=[dict(type="free", damping="0.0005")], duplicate_collision_geoms=True)

        self.upper_radius = np.array(list(map(float, self.worldbody.findall(
            "./body/body/geom")[2].items()[1][1].split())))[0]
        self.lower_radius = np.array(list(map(float, self.worldbody.findall(
            "./body/body/geom")[0].items()[1][1].split())))[0]


class DumbbellVisualObject(MujocoXMLObject):
    """
    Cup Visual object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/dumbbell-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)


class PlateObject(MujocoXMLObject):
    """
    Plate object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/plate.xml"),
                         name=name, obj_type="all", joints=[dict(type="free", damping="0.0005")], duplicate_collision_geoms=True)

        self.radius = np.array(list(map(float, self.worldbody.findall(
            "./body/body/geom")[2].items()[1][1].split())))[0]


class PlateVisualObject(MujocoXMLObject):
    """
    Plate Visual object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/plate-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)


class HollowBoxObject(MujocoXMLObject):
    """
    Hollow box object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/hollowbox.xml"),
                         name=name, obj_type="all", joints=[dict(type="free", damping="0.0005")], duplicate_collision_geoms=True)

        self.length = np.array(list(map(float, self.worldbody.findall(
            "./body/body/geom")[0].items()[1][1].split())))[0]


class HollowBoxVisualObject(MujocoXMLObject):
    """
    Hollow box Visual object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/hollowbox-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)


class CustomBoxObject(MujocoXMLObject):
    """
    Hollow box object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/custombox.xml"),
                         name=name, obj_type="all", joints=[dict(type="free", damping="0.0005")], duplicate_collision_geoms=True)

        self.length = np.array(list(map(float, self.worldbody.findall(
            "./body/body/geom")[0].items()[1][1].split())))[0]


class CustomBoxVisualObject(MujocoXMLObject):
    """
    Hollow box object (used in CompositionalEnv)
    """

    def __init__(self, name):
        super().__init__(xml_path_completion("objects/custombox-visual.xml"),
                         name=name, joints=None, obj_type="visual", duplicate_collision_geoms=True)

        self.length = np.array(list(map(float, self.worldbody.findall(
            "./body/body/geom")[0].items()[1][1].split())))[0]
