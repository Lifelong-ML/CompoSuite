import numpy as np

from composuite.arenas.trashcan_arena import TrashcanArena
from composuite.env.compositional_env import CompositionalEnv

from robosuite.utils.placement_samplers import UniformRandomSampler


class TrashcanSubtask(CompositionalEnv):
    """This class corresponds to the trashcan task for a single robot arm.
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

        self.subtask_id = 3
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
        Stages consist of reaching, grasping, lifting, hovering and dropping.

        Returns:
                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
                - (float) dropping reward
        """

        reach_mult = 0.2
        grasp_mult = 0.3
        lift_mult = 0.5
        hover_mult = 0.7
        drop_mult = 0.95

        # reaching reward governed by distance to closest object
        r_reach = 0.
        if not self.object_in_trashcan:
            # get reaching reward via minimum distance to a target object
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.object.root_body,
                target_type="body",
                return_distance=True,
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = 0
        if not self.object_in_trashcan:
            r_grasp = int(self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=[g for g in self.object.contact_geoms])
            ) * grasp_mult

        # lifting reward for picking up an object
        r_lift = 0.
        if not self.object_in_trashcan and r_grasp > 0.:
            z_target = self.trashcan_pos[2] + \
                self.trashcan_z_offset + self.trashcan_z_height + 0.2
            object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]
            z_dist = np.abs(z_target - object_z_loc)
            r_lift = grasp_mult + (1 - np.tanh(5.0 * z_dist)) * (
                lift_mult - grasp_mult
            )

        object_x_loc = self.sim.data.body_xpos[self.obj_body_id, 0]
        object_y_loc = self.sim.data.body_xpos[self.obj_body_id, 1]
        object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]

        trashcan_x_low = self.trashcan_pos[0] - self.trashcan_x_size
        trashcan_x_high = self.trashcan_pos[0] + self.trashcan_x_size
        x_check = trashcan_x_low < object_x_loc < trashcan_x_high

        trashcan_y_low = self.trashcan_pos[1] - self.trashcan_y_size
        trashcan_y_high = self.trashcan_pos[1] + self.trashcan_y_size
        y_check = trashcan_y_low < object_y_loc < trashcan_y_high

        # Not really sure what I'm doing here, so I'm copying from PickPlace
        dist = np.linalg.norm(
            np.array((self.trashcan_pos[0], self.trashcan_pos[1])
                     ) - np.array((object_x_loc, object_y_loc))
        )
        object_above_trashcan = x_check and y_check  # TODO: add z check ?

        # hover reward for getting object in front of trashcan
        r_hover = 0.
        if not self.object_in_trashcan and r_lift > 0.45:
            if object_above_trashcan:
                r_hover = lift_mult + (
                    1 - np.tanh(2.0 * dist)
                ) * (hover_mult - lift_mult)
            else:
                r_hover = r_lift + (
                    1 - np.tanh(2.0 * dist)
                ) * (hover_mult - lift_mult)

        r_drop = 0
        if not self.object_in_trashcan and object_above_trashcan and action[-1] < 0:
            r_drop = 1 * drop_mult

        # porbably add a term to encourage opening the gripper

        return r_reach, r_grasp, r_lift, r_hover, r_drop

    def not_in_trashcan(self, obj_pos):
        """Checks whether object is inside the trashcan.

        Args:
            obj_pos (np.array): Current position of the object.

        Returns:
            bool: True if the object is NOT inside the trashcan.
        """
        # get position of trashcan
        trashcan_x_low = self.trashcan_pos[0] - self.trashcan_x_size
        trashcan_x_high = self.trashcan_pos[0] + self.trashcan_x_size
        trashcan_y_low = self.trashcan_pos[1] - self.trashcan_y_size
        trashcan_y_high = self.trashcan_pos[1] + self.trashcan_y_size

        res = True
        if (
            trashcan_x_low < obj_pos[0] < trashcan_x_high
            and trashcan_y_low < obj_pos[1] < trashcan_y_high
            and self.trashcan_z_offset + self.trashcan_pos[2] < obj_pos[2] < self.trashcan_z_offset + self.trashcan_pos[2] + 0.1
        ):
            res = False
        return res

    def _get_placement_initializer(self):
        """Helper function for defining placement initializer and object sampling bounds.
        """
        super()._get_placement_initializer()

        # get limits of trashcan
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
                reference_pos=self.trashcan_pos,
                z_offset=self.trashcan_z_offset,
            )
        )

    def _load_model(self):
        """Loads an xml model, puts it in self.model
        """

        # load model for table top workspace
        self.mujoco_arena = TrashcanArena(
        )

        # settings for trashcan position
        self.trashcan_pos = self.mujoco_arena.trashcan_pos
        self.trashcan_x_size = self.mujoco_arena.trashcan_floor_size[0]
        self.trashcan_y_size = self.mujoco_arena.trashcan_floor_size[1]
        self.trashcan_z_offset = self.mujoco_arena.trashcan_floor_size[2]
        self.trashcan_z_height = self.mujoco_arena.trashcan_height
        super()._load_model()

        # Generate placement initializer
        self._initialize_model()
        self._get_placement_initializer()

    def _setup_references(self):
        """Sets up references to important components. A reference is typically an
            index or a list of indices that point to the corresponding elements
            in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        self.object_in_trashcan = False
        super()._setup_references()

    def _reset_internal(self):
        """Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_success(self):
        """Check if object has been successfully placed in the trashcan.

        Returns:
            bool: True if object is placed correctly
        """
        # remember if object is in the trashcan
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        obj_pos = self.sim.data.body_xpos[self.obj_body_id]

        self.object_in_trashcan = not self.not_in_trashcan(obj_pos)
        gripper_not_in_bin = self.not_in_trashcan(gripper_site_pos)

        if gripper_not_in_bin:
            return self.object_in_trashcan
        else:
            return 0
