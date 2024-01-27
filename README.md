# CompoSuite

This repository contains the official release of 
[CompoSuite: A Compositional Reinforcement Learning Benchmark](https://arxiv.org/pdf/2207.04136.pdf). We include pre-trained models from our CoLLAs-22 paper in our accompanying data repository, [CompoSuite-Data](https://github.com/Lifelong-ML/CompoSuite-Data), and the multi-task and compositional PPO implementations in our adaptation of Spinning Up, [CompoSuite-spinningup](https://github.com/Lifelong-ML/CompoSuite-spinningup).

CompoSuite is a benchmark of 256 compositional robotic manipulation tasks, each requiring fundamentally distinct behaviors. The diversity of tasks and their underlying compositional structure enables studying the ability of multi-task RL algorithms to extract the compositional properties of the environments, or simply explporing their ability to handle diverse tasks and generalize.

If you use any part of CompoSuite for academic research, please cite our work using the following entry:

```
@inproceedings{mendez2022composuite
  title = {Compo{S}uite: A Compositional Reinforcement Learning Benchmark},
  authors = {Jorge A. Mendez and Marcel Hussing and Meghna Gummadi and Eric Eaton},
  booktitle = {1st Conference on Lifelong Learning Agents (CoLLAs-22)},
  year = {2022}
}
```

## Installation

We provide two sets of dependencies for installing CompoSuite: `requirements_default.txt` (for installing the latest compatible version of each dependency) and `requirements_paper.txt` (for reproducing the results from our CoLLAs-22 paper). To install, execute the following command sequence:

CompoSuite requires Python<=3.11 (It might work with 3.12 but is not tested.). However, the example training scripts require Python version 3.6 or 3.7, for compatibility with Spinning Up.

```
git clone https://github.com/Lifelong-ML/CompoSuite.git
cd CompoSuite
pip install -r requirements_default.txt
pip install -e .
```

To use the example training scripts, we provide a slightly modified version of Spinning Up which contains some minor changes to functionalities as well as the compositional learner, as described in our paper. This step is only required for running the example training scripts. To install our Spinning Up version, run:

```
git clone https://github.com/Lifelong-ML/CompoSuite-spinningup.git
cd spinningup
pip install -e .
```


## Using CompoSuite

To create an individual CompoSuite environment, execute:

```
composuite.make(ROBOT_NAME, OBJECT_NAME, OBSTACLE_NAME, OBJECTIVE_NAME, use_task_id_obs=False)
```

This returns a Gym environment that can be directly used for single-task training. The `use_task_id_obs` flag is used internally by CompoSuite to determine whether the observation should include a multi-hot indicator of the components that make up the task, as described in the paper. In the example above, we have set it to `False` because single-task agents make no use of the components.

CompoSuite supports three problem settings in terms of how tasks are sampled, as described below: the full CompoSuite benchmark, small-scale CompoSuite&cap;&#60;element&#62; benchmarks, and restricted sampling CompoSuite&#92;&#60;element&#62; benchmarks. Example code for how to train using Spinning Up under each of those settings, as well as a single-task setting, are included in `composuite/example_scripts/`. For multi-task training, make sure to set `use_task_id_obs=True`, unless you wish to withhold this information (note that with the default observation space, this is required to distinguish between tasks with a single observation).

### Full CompoSuite

In this setting, tasks are sampled uniformly at random from all possible CompoSuite tasks. The following command creates two lists of tasks, one for training and one for testing zero-shot generalization:

```
train_tasks, test_tasks = composuite.sample_tasks(experiment_type='default', num_train=224)
```

Each task in the lists is a tuple `(ROBOT_NAME, OBJECT_NAME, OBSTACLE_NAME, OBJECTIVE_NAME)` that can be used to create a Gym environment with `composuite.make(...)`.

### Small-scale CompoSuite&cap;&#60;element&#62;

This setting restricts the sampling to the set of tasks that contain the &#60;element&#62;. For example, CompoSuite&cap;IIWA considers only tasks that contain the IIWA arm. This restricts the sample to only 64 tasks, and limits the diversity of the tasks, both of which are useful during development. There are 16 possible choices of &#60;element&#62;, leading to 16 benchmarks of 64 tasks each. To sample training and evaluation tasks in this setting, execute:

```
train_tasks, test_tasks = composuite.sample_tasks(experiment_name='smallscale', smallscale_elem=<element>, num_train=32)
```

### Restricted CompoSuite&#92;&#60;element&#62;

In this last setting, the training set includes a single task with &#60;element&#62; and multiple uniformly sampled tasks without &#60;element&#62;, and the test set contains only the remaining tasks with &#60;element&#62;. This is the most challenging CompoSuite setting; for example, the agent might be required to solve all IIWA tasks after having trained on a single IIWA task. The following command creates the training and test set of tasks for this setting:

```
train_tasks, test_tasks = composuite.sample_tasks(experiment_name='holdout', holdout_elem=<element>, num_train=56)
```

## Using our example training scripts

To train on a single environment using our example training scripts, run:

```
python -m example_scripts.train_ppo
```

To run something with multiple environments, the example scripts use MPI. For example, for training on 4 tasks with 4 MPI processes, run:

```
mpirun -np 4 python -m example_scripts.train_ppo --experiment-type smallscale --smallscale-elem IIWA --learner-type comp --num-train 4
```

Note that the multi-task and compositional PPO implementations in CompoSuite-spinningup assume that there is one MPI process per CompoSuite task (i.e., set `-np x` and `--num-train x` where `x` is the number of tasks).
