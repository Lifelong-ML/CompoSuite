def choose_object():
    """
    Prints out object options, and returns the requested object.

    Returns:
        str: Requested object name
    """
    # Get the list of objects
    objects = {
        "Box",
        "Dumbbell",
        "Plate",
        "Hollowbox"
    }

    # Make sure set is deterministically sorted
    objects = sorted(objects)

    # Select object
    print("Here is a list of available objects:\n")

    for k, obj in enumerate(objects):
        print("[{}] {}".format(k, obj))
    print()
    try:
        s = input(
            "Choose an object "
            + "(enter a number from 0 to {}): ".format(len(objects) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(objects))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(
            list(objects)[k]))

    # Return requested object
    return list(objects)[k]


def choose_obstacle():
    """
    Prints out obstacle options, and returns the requested obstacle.

    Returns:
        str: Requested obstacle name
    """
    # Get the list of obstacles
    obstacles = {
        "None",
        "ObjectWall",
        "GoalWall",
        "ObjectDoor",
    }

    # Make sure set is deterministically sorted
    obstacles = sorted(obstacles)

    # Select obstacle
    print("Here is a list of available obstacles:\n")

    for k, obstacle in enumerate(obstacles):
        print("[{}] {}".format(k, obstacle))
    print()
    try:
        s = input(
            "Choose an obstacle "
            + "(enter a number from 0 to {}): ".format(len(obstacles) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(obstacles))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(
            list(obstacles)[k]))

    # Return requested obstacle
    return list(obstacles)[k]


def choose_task():
    """
    Prints out task options, and returns the requested task.

    Returns:
        str: Requested task name
    """
    # Get the list of tasks
    tasks = {
        "PickPlace",
        "Shelf",
        "Push",
        "Trashcan"
    }

    # Make sure set is deterministically sorted
    tasks = sorted(tasks)

    # Select task
    print("Here is a list of available tasks:\n")

    for k, task in enumerate(tasks):
        print("[{}] {}".format(k, task))
    print()
    try:
        s = input(
            "Choose an task "
            + "(enter a number from 0 to {}): ".format(len(tasks) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(tasks))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(list(tasks)[k]))

    # Return requested task
    return list(tasks)[k]

def choose_robots(exclude_bimanual=False):
    """
    Prints out robot options, and returns the requested robot. Restricts options to single-armed robots if
    @exclude_bimanual is set to True (False by default)

    Args:
        exclude_bimanual (bool): If set, excludes bimanual robots from the robot options

    Returns:
        str: Requested robot name
    """
    # Get the list of robots
    robots = {
        "Panda",
        "Jaco",
        "Kinova3",
        "IIWA",
    }

    # Add Baxter if bimanual robots are not excluded
    if not exclude_bimanual:
        robots.add("Baxter")

    # Make sure set is deterministically sorted
    robots = sorted(robots)

    # Select robot
    print("Here is a list of available robots:\n")

    for k, robot in enumerate(robots):
        print("[{}] {}".format(k, robot))
    print()
    try:
        s = input(
            "Choose a robot "
            + "(enter a number from 0 to {}): ".format(len(robots) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(robots))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(list(robots)[k]))

    # Return requested robot
    return list(robots)[k]