import numpy as np

from composuite.arenas.pick_place_arena import PickPlaceArena
from composuite.env.compositional_env import CompositionalEnv

import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler


class PickPlaceSubtask(CompositionalEnv):
    """This class corresponds to the pick place task for a single robot arm.
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

        self.subtask_id = 0
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
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.2
        grasp_mult = 0.3
        lift_mult = 0.5
        hover_mult = 0.7
        drop_mult = 0.9

        # reaching reward governed by distance to closest object
        r_reach = 0.
        if not self.object_in_bin:
            # get reaching reward via minimum distance to a target object
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.object.root_body,
                target_type="body",
                return_distance=True,
            )

            r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # grasping reward for touching any objects of interest
        is_grasping = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.object.contact_geoms])
        r_grasp = int(is_grasping) * grasp_mult

        # lifting reward for picking up an object
        r_lift = 0.
        if not self.object_in_bin and r_grasp > 0.:
            z_target = self.bin2_pos[2] + 0.25
            object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]
            z_dist = np.abs(z_target - object_z_loc)
            r_lift = grasp_mult + (1 - np.tanh(5.0 * z_dist)) * (
                lift_mult - grasp_mult
            )

        # segment objects into left of the bins and above the bins
        object_xy_loc = self.sim.data.body_xpos[self.obj_body_id, :2]
        y_check = (
            np.abs(object_xy_loc[1] -
                   self.bin2_pos[1])
            < self.bin2_size[1]
        )
        x_check = (
            np.abs(object_xy_loc[0] -
                   self.bin2_pos[0])
            < self.bin2_size[0]
        )
        object_above_bin = x_check and y_check

        # hover reward for getting object above bin
        r_hover = 0.
        r_drop = 0.
        if not self.object_in_bin and r_lift > 0.45:
            dist = np.linalg.norm(
                self.bin2_pos[:2] - object_xy_loc
            )
            # objects to the left get r_lift added to hover reward,
            # those on the right get max(r_lift) added (to encourage dropping)
            if not object_above_bin:
                r_hover = r_lift + (
                    1 - np.tanh(2.0 * dist)
                ) * (hover_mult - lift_mult)
            else:
                r_hover = lift_mult + (
                    1 - np.tanh(2.0 * dist)
                ) * (hover_mult - lift_mult)

        if r_grasp > 0 and object_above_bin:
            z_target = self.bin2_pos[2] + 0.1
            object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]
            z_dist = np.maximum(object_z_loc - z_target, 0.)
            r_drop = hover_mult + \
                (1 - np.tanh(5.0 * z_dist)) * (drop_mult - hover_mult)

        return r_reach, r_grasp, r_lift, r_hover, r_drop

    def not_in_bin(self, obj_pos):
        """Checks whether object is in the second bin.

        Args:
            obj_pos (np.array): Current position of the object.

        Returns:
            bool: True if the object is NOT inside the second bin.
        """
        bin_x_low = self.bin2_pos[0] - self.bin2_size[0]
        bin_y_low = self.bin2_pos[1] - self.bin2_size[1]

        bin_x_high = self.bin2_pos[0] + self.bin2_size[0]
        bin_y_high = self.bin2_pos[1] + self.bin2_size[1]

        res = True
        if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and self.bin2_pos[2] < obj_pos[2] < self.bin2_pos[2] + 0.1
        ):
            res = False
        return res

    def _get_placement_initializer(self):
        """Helper function for defining placement initializer 
           and object sampling bounds. Extends super class initilizer.
        """
        super()._get_placement_initializer()

        # TODO: why is this not exactly in the middle
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
                z_offset=self.bin2_pos[2] - self.bin1_pos[2],
            )
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """

        # load model for table top workspace
        self.mujoco_arena = PickPlaceArena(
            bin1_pos=self.bin1_pos,
        )

        # Load model propagation
        super()._load_model()

        # Generate placement initializer
        self._initialize_model()
        self._get_placement_initializer()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """

        # keep track of which objects are in their corresponding bins
        self.object_in_bin = False

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((1, 3))

        # TODO: fix this once i understand why its here
        # I think we can remove target bin placements
        bin_x_low = self.bin2_pos[0]
        bin_y_low = self.bin2_pos[1]
        bin_x_low += self.bin2_size[0] / 2.
        bin_y_low += self.bin2_size[1] / 2.
        self.target_bin_placements[0, :] = [
            bin_x_low, bin_y_low, self.bin2_pos[2]]

        super()._setup_references()

    def _reset_internal(self):
        """Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Set the bins to the desired position
        self.sim.model.body_pos[self.sim.model.body_name2id(
            "bin1")] = self.bin1_pos
        self.sim.model.body_pos[self.sim.model.body_name2id(
            "bin2")] = self.bin2_pos

    def _check_success(self):
        """
        Check if all objects have been successfully placed in their corresponding bins.

        Returns:
            bool: True if object is placed correctly
        """
        # remember objects that are in the correct bins
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        obj_pos = self.sim.data.body_xpos[self.obj_body_id]
        dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = 1 - np.tanh(10.0 * dist)
        # self.object_in_bin = not self.not_in_bin(obj_pos)
        self.object_in_bin = bool(
            (not self.not_in_bin(obj_pos)) and r_reach > 0.35)

        return self.object_in_bin

