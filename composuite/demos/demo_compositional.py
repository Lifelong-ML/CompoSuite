import composuite
from composuite.utils.demo_utils import *
from robosuite.utils.input_utils import *


def simulate_compositional_env():
    """Initialize a compositional environment and run with
       random actions for visualization.
    """
    robot = choose_robots(exclude_bimanual=True)
    task = choose_task()
    obj = choose_object()
    obstacle = choose_obstacle()
    if obstacle == "None": obstacle = None

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = composuite.make(
        robot=robot,
        obj=obj,
        obstacle=obstacle,
        task=task,
        has_renderer=True,
        ignore_done=True,
    )
    env.reset()
    env.viewer.set_camera(camera_id=1)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for _ in range(10000):
        action = np.random.uniform(low, high)
        env.step(action)
        env.render()

if __name__ == "__main__":
    simulate_compositional_env()
