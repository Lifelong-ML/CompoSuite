import numpy as np
import argparse
import os
import json
import warnings

import torch

import composuite

from spinup.algos.pytorch.ppo.core import MLPActorCritic
from spinup.algos.pytorch.ppo.compositional_core import CompositionalMLPActorCritic
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_tools import proc_id


def parse_default_args():
    parser = argparse.ArgumentParser()

    # Directory information
    parser.add_argument('--data-dir', default='spinningup_training')
    parser.add_argument('--load-dir', default=None)

    # Neural network training parameters
    parser.add_argument('--exp-name', type=str, default='ppo')
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--steps', type=int, default=16000)
    parser.add_argument('--epochs', type=int, default=625)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--pi-lr', type=float, default=1e-4)
    parser.add_argument('--vf-lr', type=float, default=1e-4)
    parser.add_argument('--pi-iters', type=int, default=128)
    parser.add_argument('--vf-iters', type=int, default=128)
    parser.add_argument('--target-kl', type=float, default=0.02)
    parser.add_argument('--ent-coef', type=float, default=0.)
    parser.add_argument('--log-std-init', type=float, default=0.)

    # Architecture information
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)

    # Task information
    parser.add_argument('--controller', type=str, default="joint")

    parser.add_argument('--robot', type=str, default="IIWA",
                        choices=['IIWA', 'Jaco', 'Kinova3', 'Panda'],
                        help='Robotic manipulator to use in task.')
    parser.add_argument('--object', type=str, default="Hollowbox",
                        choices=['Hollowbox', 'Box', 'Dumbbell', 'Plate'],
                        help='Object to use in the environment.')
    parser.add_argument('--obstacle', type=str, default='None',
                        choices=['None', 'ObjectDoor', 
                                 'ObjectWall', 'Goalwall'],
                        help='Obstacle to use in environment.')
    parser.add_argument('--task', type=str, default="PickPlace",
                        choices=['PickPlace', 'Push', 'Shelf', 'Trashcan'],
                        help='Objective to train on.')
    parser.add_argument('--horizon', type=int, default=500,
                        help='Number of steps before the environment resets')
    parser.add_argument('--task-id', type=int, default=-1,
                        help='This ID is used for the single task \
                              learner as an unravel index \
                              into the possible task configurations. \
                              It can take a value from 0 to 255. The \
                              default of -1 means that it is not used. \
                              If this is used, the args for robot, \
                              object, task and obstacle will be \
                              overwritten by the specific configuration. \
                              This main purpose of the flag is to run all \
                              possible single task experiments by running \
                              all indices from 0 to 255.')
    parser.add_argument('--num-train-tasks', type=int, default=1)

    # Learner type
    parser.add_argument('--learner-type', type=str, default='stl',
                        choices=['stl', 'mtl', 'comp'],
                        help='stl: Single-task learner, \
                              mtl: Multi-task learner, \
                              comp: Compositional learner')

    # Experiment type
    parser.add_argument('--experiment-type', default='default',
                        choices=['default', 'smallscale', 'holdout'])
    parser.add_argument('--smallscale-elem', default='IIWA')
    parser.add_argument('--holdout-elem', default='IIWA')

    args = parser.parse_args()

    if args.experiment_type == 'default':
        args.exp_name = os.path.join(
            args.exp_name, args.experiment_type +
            "_" + str(args.num_train_tasks),
            args.learner_type, 's' + str(args.seed))
    elif args.experiment_type == 'holdout':
        args.exp_name = os.path.join(
            args.exp_name, args.experiment_type +
            "_" + str(args.num_train_tasks),
            args.learner_type, args.holdout_elem, 's' + str(args.seed))
    elif args.experiment_type == 'smallscale':
        args.exp_name = os.path.join(
            args.exp_name, args.experiment_type +
            "_" + str(args.num_train_tasks),
            args.learner_type, args.smallscale_elem, 's' + str(args.seed))

    assert args.task_id >= -1 and args.task_id <= 255, "Task ID out of bounds"
    if args.learner_type == 'stl':
        assert args.experiment_type == 'default', \
            "You have selected the single-task agent but are \
                        trying to run a multi-task experiment."

    if args.task_id != -1 and args.learner_type != 'stl':
        warnings.warn("You have selected a specific task ID while using \
                       a learner that is not the single-task learner.  \
                       Ignoring task ID.")
    return args


def train_model(robot, obj, obstacle, task, args, logger_kwargs):
    """Run a single model training.

    Args:
        robot (str): Robot to use in environment.
        obj (str): Object to use in environment.
        obstacle (str): Obstacle to use in environment.
        task (str): Objective to use in environment.
        args (Namespace): Training namespace arguments.
        logger_kwargs (dict): Additional kwargs for spinningup ppo.

    Raises:
        NotImplementedError: Raises if unknown 
                             learner type is selected.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    checkpoint = None
    if args.load_dir is not None:
        checkpoint = torch.load(os.path.join(
            args.load_dir, 'pyt_save', 'state_dicts.pt'))

    def env_fn():
        return composuite.make(
            robot, obj, obstacle, task,
            args.controller, args.horizon, use_task_id_obs=True)

    if args.learner_type == 'comp':
        network_class = CompositionalMLPActorCritic
        hidden_sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))

        ac_kwargs = {
            'log_std_init': args.log_std_init,
            'hidden_sizes': hidden_sizes,
            'module_names': ['obstacle_id', 'object_id',
                             'subtask_id', 'robot_id'],
            'module_input_names': ['obstacle-state', 'object-state',
                                   'goal-state', 'robot0_proprio-state'],
            'interface_depths': [-1, 1, 2, 3],
            'graph_structure': [[0], [1], [2], [3]],
        }
    elif args.learner_type == 'stl' or args.learner_type == 'mtl':
        network_class = MLPActorCritic
        ac_kwargs = dict(
            hidden_sizes=[args.hid]*args.l, log_std_init=args.log_std_init)
    else:
        raise NotImplementedError("Unknown learner type was selected. \
                                   Options are stl, mtl and comp.")

    ppo(env_fn=env_fn,
        actor_critic=network_class,
        ac_kwargs=ac_kwargs,
        seed=args.seed, gamma=args.gamma, steps_per_epoch=args.steps,
        epochs=args.epochs, clip_ratio=args.clip,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr,
        train_pi_iters=args.pi_iters, train_v_iters=args.vf_iters,
        target_kl=args.target_kl, logger_kwargs=logger_kwargs,
        max_ep_len=args.horizon, ent_coef=args.ent_coef,
        log_per_proc=args.learner_type != 'stl', checkpoint=checkpoint)


def main():
    """Runs the training script.

    Raises:
        NotImplementedError: Raises if selected 
                             experiment type does not exist.
    """
    args = parse_default_args()
    np.random.seed(args.seed)
    if args.experiment_type == 'default':
        if args.learner_type == 'stl':
            if args.task_id != -1:
                train_tasks, test_tasks = composuite.sample_tasks(
                    experiment_type=args.experiment_type,
                    num_train=0,
                    no_shuffle=True
                )
                train_task = test_tasks[args.task_id]
                robot, obj, obstacle, task = train_task
                test_tasks.remove(train_task)
                train_tasks.append(train_task)
            else:
                robot = args.robot
                obj = args.object
                task = args.task
                obstacle = args.obstacle

                train_tasks = [(robot, obj, obstacle, task)]
                _, test_tasks = composuite.sample_tasks(
                    experiment_type=args.experiment_type,
                    num_train=0,
                    no_shuffle=True
                )
                test_tasks.remove(train_tasks[0])
        else:
            train_tasks, test_tasks = composuite.sample_tasks(
                experiment_type=args.experiment_type,
                num_train=args.num_train_tasks,
                shuffling_seed=args.seed,
                no_shuffle=False
            )
            robot, obj, obstacle, task = train_tasks[proc_id()]
    elif args.experiment_type == 'smallscale' or \
            args.experiment_type == 'holdout':
        train_tasks, test_tasks = composuite.sample_tasks(
            experiment_type=args.experiment_type,
            num_train=args.num_train_tasks,
            smallscale_elem=args.smallscale_elem,
            holdout_elem=args.holdout_elem,
            shuffling_seed=args.seed,
            no_shuffle=False
        )
        robot, obj, obstacle, task = train_tasks[proc_id()]
    else:
        raise NotImplementedError("Specified experiment type does not exist.")

    args.robot = robot
    args.object = obj
    args.obstacle = obstacle
    args.task = task

    os.makedirs(os.path.join(args.data_dir, args.exp_name), exist_ok=True)
    with open(
        os.path.join(
            args.data_dir, args.exp_name, 'args_{}.json'.format(proc_id())),
            'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if proc_id() == 0:
        with open(os.path.join(args.data_dir, args.exp_name, 'tasks.json'), 'w') as f:
            tasks = {
                'train_tasks': train_tasks,
                'test_tasks': test_tasks
            }
            json.dump(tasks, f, indent=2)

    logger_kwargs = setup_logger_kwargs(
        args.exp_name, data_dir=args.data_dir)

    train_model(
        robot=robot, obj=obj, obstacle=obstacle, task=task,
        args=args, logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    main()
