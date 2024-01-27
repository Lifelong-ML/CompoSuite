from cgitb import small
import warnings
import itertools
import random

from robosuite.controllers import load_controller_config
from composuite.env.gym_wrapper import GymWrapper
import robosuite as suite

AVAILABLE_ROBOTS = ["IIWA", "Jaco", "Kinova3", "Panda"]
AVAILABLE_OBSTACLES = [None, "None", "GoalWall",
                       "ObjectDoor", "ObjectWall"]
AVAILABLE_OBJECTS = ["Box", "Dumbbell", "Plate", "Hollowbox"]
AVAILABLE_TASKS = ["PickPlace", "Push", "Shelf", "Trashcan"]


def make(robot="IIWA", obj="Box", obstacle=None, 
         task="PickPlace", controller="joint", env_horizon=500, 
         has_renderer=False, has_offscreen_renderer=False, 
         reward_shaping=True, ignore_done=True, use_camera_obs=False, 
         **kwargs) -> GymWrapper:
    """Create a compositional environment in form of a gym environment.

    Args:
        robot (str, optional): Robot to use in environment. 
                               Defaults to "IIWA".
        obj (str, optional): Object to use in environment. 
                             Defaults to "milk".
        obstacle (_type_, optional): Obstacle to use in environment. 
                                     Defaults to None.
        task (str, optional): Objective to use in environment. 
                              Defaults to "PickPlace".
        controller (str, optional): Robot controller. 
                                    Options are osc, joint 
                                    and osc_pose. Defaults to "osc".
        env_horizon (int, optional): Number of steps after which env resets. 
                                     Defaults to 500.
        has_renderer (bool, optional): True, if environment should have a renderer
                                       to support visualization. 
                                       Defaults to False.
        has_offscreen_renderer (bool, optional): True, if environment 
                                                 should have an offscreen renderer
                                                 to support visual observations. 
                                                 Defaults to False.
        reward_shaping (bool, optional): True, if shaped rewards instead of sparse
                                         rewards should be used. Defaults to True.
        ignore_done (bool, optional): True, if environment should not output done
                                      after reaching horizon. Defaults to True.
        use_camera_obs (bool, optional): True, if environment should return visual
                                         observations. Defaults to False.

    Raises:
        NotImplementedError: Raises if environment name is misspecified.
        NotImplementedError: Raises if controller is misspecified.

    Returns:
        GymWrapper: Gym-like environment
    """

    assert robot in AVAILABLE_ROBOTS
    assert obstacle in AVAILABLE_OBSTACLES
    assert obj in AVAILABLE_OBJECTS
    assert task in AVAILABLE_TASKS

    if obstacle == "None":
        obstacle = None

    # defined options to create robosuite environment
    options = {}

    if task == "PickPlace":
        options["env_name"] = "PickPlaceSubtask"
    elif task == "Push":
        options["env_name"] = "PushSubtask"
    elif task == "Shelf":
        options["env_name"] = "ShelfSubtask"
    elif task == "Trashcan":
        options["env_name"] = "TrashcanSubtask"
    else:
        raise NotImplementedError

    options["robots"] = robot
    options["obstacle"] = obstacle
    options["object_type"] = obj

    if controller == "osc":
        controller_name = "OSC_POSITION"
    elif controller == "joint":
        controller_name = "JOINT_POSITION"
    elif controller == 'osc_pose':
        controller_name = 'OSC_POSE'
    else:
        print("Controller unknown")
        raise NotImplementedError

    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)

    env = suite.make(
        **options,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        reward_shaping=reward_shaping,
        ignore_done=ignore_done,
        use_camera_obs=use_camera_obs,
        horizon=env_horizon,
        **kwargs
    )

    env.reset()
    return GymWrapper(env)


def sample_tasks(experiment_type='default', num_train=1,
                 smallscale_elem=None, holdout_elem=None,
                 shuffling_seed=None, no_shuffle=False):
    """Sample a set of training and test configurations as tasks.

    Args:
        experiment_type (str, optional): The type of experiment that
            is used for sampling. The possible experiments are the experiments
            referred to in the initial publication. 
            Options are: default, smallscale, holdout. Defaults to 'default'.
        num_train (int, optional): Number of training tasks to sample. 
            All other tasks will be returned as test tasks. Defaults to 1.
        smallscale_elem (_type_, optional): Required when experiment_type
            smallscale. The axis element that is to be fixed. No other element
            from that axis will be sampled. Defaults to None.
        holdout_elem (_type_, optional): Required when experiment_type is
            holdout. Only a single configuration containing this element will
            be returned in the train set. The test set contains all other tasks
            using this element other than the single one in the train set.
            Defaults to None.
        shuffling_seed (_type_, optional): Random seed for shuffling configurations. 
            This will reset to the state before shuffling so that your random 
            seed selection outside this method is not affected.
            Defaults to None.

    Raises:
        NotImplementedError: Raises if experiment type is unkown.
        NotImplementedError: Raises if selected 
            axis element in non-default experiments is unknown.

    Returns:
        List[Tuple]: List containing the training task configurations.
        List[Tuple]: List containing the test task configurations.
    """
    if experiment_type == 'smallscale':
        assert num_train >= 0 and num_train <= 64, \
            "Number of tasks must be at least 1 and at \
             most 64 in smallscale setting."
    elif experiment_type == 'holdout':
        assert num_train >= 1 and num_train <= 193, \
            "Number of tasks must be at least 1 and at \
             most 193 in holdout setting."
    elif experiment_type == 'default':
        assert num_train >= 0 and num_train <= 256, \
            "Number of tasks must be at least 1 and at \
             most 256 in default setting"
    else:
        raise NotImplementedError("Specified experiment type does not exist. \
            Options are: default, smallscale, holdout.")

    if experiment_type == 'default':
        if smallscale_elem is not None or holdout_elem is not None:
            warnings.warn("You specified a smallscale/holdout element but are \
                using the default experiment setting. Continuing and ignoring \
                smallscale/holdout element.")
        elif experiment_type == 'smallscale':
            assert smallscale_elem is not None, \
                "Selected experiment type smallscale \
                 but did not specifiy fixed element."
        elif experiment_type == 'holdout':
            assert holdout_elem is not None, \
                "Selected experiment type holdout \
                 but did not specifiy single task element."

    if experiment_type == 'default' or experiment_type == 'holdout':
        _robots = AVAILABLE_ROBOTS
        _objects = AVAILABLE_OBJECTS
        _obstacles = [e for e in AVAILABLE_OBSTACLES if e is not None]
        _objectives = AVAILABLE_TASKS
    elif experiment_type == 'smallscale':
        _robots = AVAILABLE_ROBOTS
        _objects = AVAILABLE_OBJECTS
        _obstacles = [e for e in AVAILABLE_OBSTACLES if e is not None]
        _objectives = AVAILABLE_TASKS

        if smallscale_elem in AVAILABLE_ROBOTS:
            _robots = [smallscale_elem]
        elif smallscale_elem in AVAILABLE_OBJECTS:
            _objects = [smallscale_elem]
        elif smallscale_elem in AVAILABLE_OBSTACLES:
            _obstacles = [smallscale_elem]
        elif smallscale_elem in AVAILABLE_TASKS:
            _objectives = [smallscale_elem]
        else:
            raise NotImplementedError(
                "Specified smallscale element does not exist. Options are: \n\
                IIWA, Jaco, Kinova3, Panda, None, GoalWall, \
                ObjectDoor, ObjectWall, Box, Dumbbell, Plate, Hollowbox \
                PickPlace, Push, Shelf, Trashcan")

    all_configurations = []
    for conf in itertools.product(_robots, _objects, _obstacles, _objectives):
        all_configurations.append(conf)

    state = random.getstate()
    if shuffling_seed is not None:
        random.seed(shuffling_seed)
    if not no_shuffle:
        random.shuffle(all_configurations)
    random.setstate(state)

    if experiment_type == 'default' or experiment_type == 'smallscale':
        return all_configurations[:num_train], all_configurations[num_train:]
    elif experiment_type == 'holdout':
        elem_tasks = [
            conf for conf in all_configurations if holdout_elem in conf]
        non_elem_tasks = [
            conf for conf in all_configurations if holdout_elem not in conf]

        train_elem_task = elem_tasks[:1]
        holdout_tasks = elem_tasks[1:]

        return non_elem_tasks[:num_train - 1] + train_elem_task, holdout_tasks
