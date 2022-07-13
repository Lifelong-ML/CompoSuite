import numpy as np

from composuite.arenas.push_arena import PushArena
from composuite.env.compositional_env import CompositionalEnv

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class PushSubtask(CompositionalEnv):
    """This class corresponds to the lifting task for a single robot arm.
    """

    def __init__(
        self,
        robots,
        object_type,
        obstacle,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="RethinkGripper",
        initialization_noise=None,
        use_camera_obs=True,
        use_object_obs=True,
        use_task_id_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        bin1_pos=(0.1, -0.26, 0.8),
        bin2_pos=(0.1, 0.13, 0.8),
        reward_scale=1.0,
        reward_shaping=False,
    ):

        self.subtask_id = 1
        super().__init__(
            robots,
            object_type,
            obstacle,
            bin1_pos,
            bin2_pos,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            use_task_id_obs=use_task_id_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
        )

    def staged_rewards(self, action):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping and approaching.

        Returns:
                - (float) reaching reward
                - (float) grasping reward
                - (float) approach reward
        """
        reach_mult = 0.2
        grasp_mult = 0.3
        approach_mult = 0.7

        # reaching reward governed by distance to object
        obj_pos = self.sim.data.body_xpos[self.obj_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # touch reward for grasping the object
        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.object.contact_geoms])
        ) * grasp_mult

        r_approach = 0.
        if r_grasp > 0:
            # approach reward for approaching goal with object
            goal_pos = self.sim.model.body_pos[self.goal_body_id]
            dist = np.linalg.norm(goal_pos - obj_pos)
            r_approach = grasp_mult + \
                (1 - np.tanh(5. * dist)) * (approach_mult - grasp_mult)

        return r_reach, r_grasp, r_approach

    def _load_model(self):
        """Loads an xml model, puts it in self.model
        """
        # Load model propagation

        # load model for table top workspace
        self.mujoco_arena = PushArena(
            bin1_pos=self.bin1_pos,
            bin2_pos=self.bin2_pos,
        )

        super()._load_model()
        self._initialize_model()
        self._get_placement_initializer()

    def _get_placement_initializer(self):
        """Helper function for defining placement initializer and object sampling bounds.
        """
        super()._get_placement_initializer()

        # placement is relative to object bin, so compute difference and send to placement initializer
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"{self.visual_object.name}ObjectSampler",
                mujoco_objects=self.visual_object,
                x_range=[0, 0],
                y_range=[0, 0],
                rotation=0.,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.bin2_pos,
                z_offset=0.,
            )
        )

    def _setup_references(self):
        """Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """

        super()._setup_references()

    def _reset_internal(self):
        """Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_success(self):
        """Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        obj_xy_pos = self.sim.data.body_xpos[self.obj_body_id][:2]
        goal_xy_pos = self.sim.model.body_pos[self.goal_body_id][:2]

        obj_height = self.sim.data.body_xpos[self.obj_body_id][2]
        table_height = self.bin2_pos[2]

        return np.linalg.norm(obj_xy_pos - goal_xy_pos) <= 0.03 and obj_height <= table_height + 0.1

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method

        """
        reward, _, info = super()._post_action(action)

        object_height = self.sim.data.body_xpos[self.obj_body_id][2]
        table_height = self.bin2_pos[2]

        self.done = object_height - \
            table_height > 0.1 or (
                (self.timestep >= self.horizon) and not self.ignore_done)

        return reward, self.done, info
