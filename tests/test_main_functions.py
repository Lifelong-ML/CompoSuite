import pytest
from composuite import sample_tasks

AVAILABLE_ROBOTS = ["IIWA", "Jaco", "Kinova3", "Panda"]
AVAILABLE_OBSTACLES = ["None", "GoalWall",
                       "ObjectDoor", "ObjectWall"]
AVAILABLE_OBJECTS = ["Box", "Dumbbell", "Plate", "Hollowbox"]
AVAILABLE_TASKS = ["PickPlace", "Push", "Shelf", "Trashcan"]


def test_sample_tasks_smallscale():
    possible_elements = AVAILABLE_ROBOTS + AVAILABLE_OBSTACLES + \
        AVAILABLE_OBJECTS + AVAILABLE_TASKS
    for e in possible_elements:
        train_tasks, test_tasks = sample_tasks(
            experiment_type='smallscale', num_train=32, smallscale_elem=e
        )

        for t in train_tasks:
            assert e in t
        for t in test_tasks:
            assert e in t


def test_sample_tasks_holdout():
    possible_elements = AVAILABLE_ROBOTS + AVAILABLE_OBSTACLES + \
        AVAILABLE_OBJECTS + AVAILABLE_TASKS
    for e in possible_elements:
        train_tasks, test_tasks = sample_tasks(
            experiment_type='holdout', num_train=56, holdout_elem=e
        )
        assert len(train_tasks) == 56
        assert len(test_tasks) == 63

        num_train_tasks_with_e = 0
        for t in train_tasks:
            if e in t:
                num_train_tasks_with_e += 1
        assert num_train_tasks_with_e == 1

        for t in test_tasks:
            assert e in t


def test_sample_tasks_num_train():
    num_train = 0
    train_tasks, test_tasks = sample_tasks(num_train=num_train)
    assert len(train_tasks) == num_train
    assert len(test_tasks) == 256 - num_train

    num_train = 5
    train_tasks, test_tasks = sample_tasks(num_train=num_train)
    assert len(train_tasks) == num_train
    assert len(test_tasks) == 256 - num_train

    num_train = 256
    train_tasks, test_tasks = sample_tasks(num_train=num_train)
    assert len(train_tasks) == num_train
    assert len(test_tasks) == 256 - num_train

    with pytest.raises(AssertionError):
        num_train = -1
        train_tasks, test_tasks = sample_tasks(num_train=num_train)

    with pytest.raises(AssertionError):
        num_train = 257
        train_tasks, test_tasks = sample_tasks(num_train=num_train)


def test_sample_tasks_num_train_smallscale_1():
    possible_elements = AVAILABLE_ROBOTS + AVAILABLE_OBSTACLES + \
        AVAILABLE_OBJECTS + AVAILABLE_TASKS
    num_train = 0
    for e in possible_elements:
        train_tasks, test_tasks = sample_tasks(
            experiment_type='smallscale', num_train=num_train, smallscale_elem=e
        )
        assert len(train_tasks) == num_train
        assert len(test_tasks) == 64 - num_train


def test_sample_tasks_num_train_smallscale_2():
    possible_elements = AVAILABLE_ROBOTS + AVAILABLE_OBSTACLES + \
        AVAILABLE_OBJECTS + AVAILABLE_TASKS
    num_train = -1
    for e in possible_elements:
        with pytest.raises(AssertionError):
            train_tasks, test_tasks = sample_tasks(
                experiment_type='smallscale', num_train=num_train, smallscale_elem=e
            )

    num_train = 65
    for e in possible_elements:
        with pytest.raises(AssertionError):
            train_tasks, test_tasks = sample_tasks(
                experiment_type='smallscale', num_train=num_train, smallscale_elem=e
            )


def test_sample_tasks_num_train_holdout_1():
    possible_elements = AVAILABLE_ROBOTS + AVAILABLE_OBSTACLES + \
        AVAILABLE_OBJECTS + AVAILABLE_TASKS
    num_train = 0
    for e in possible_elements:
        with pytest.raises(AssertionError):
            train_tasks, test_tasks = sample_tasks(
                experiment_type='holdout', num_train=num_train, holdout_elem=e
            )
