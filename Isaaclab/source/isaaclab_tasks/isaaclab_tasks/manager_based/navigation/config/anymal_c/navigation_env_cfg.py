# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# import math

# from isaaclab.envs import ManagerBasedRLEnvCfg
# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import ObservationGroupCfg as ObsGroup
# from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers import TerminationTermCfg as DoneTerm
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# import isaaclab_tasks.manager_based.navigation.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg

# LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_base = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
#             "velocity_range": {
#                 "x": (-0.0, 0.0),
#                 "y": (-0.0, 0.0),
#                 "z": (-0.0, 0.0),
#                 "roll": (-0.0, 0.0),
#                 "pitch": (-0.0, 0.0),
#                 "yaw": (-0.0, 0.0),
#             },
#         },
#     )


# @configclass
# class ActionsCfg:
#     """Action terms for the MDP."""

#     pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
#         asset_name="robot",
#         policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
#         low_level_decimation=4,
#         low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
#         low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
#     )


# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         # observation terms (order preserved)
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
#         projected_gravity = ObsTerm(func=mdp.projected_gravity)
#         pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
#     position_tracking = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.5,
#         params={"std": 2.0, "command_name": "pose_command"},
#     )
#     position_tracking_fine_grained = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.5,
#         params={"std": 0.2, "command_name": "pose_command"},
#     )
#     orientation_tracking = RewTerm(
#         func=mdp.heading_command_error_abs,
#         weight=-0.2,
#         params={"command_name": "pose_command"},
#     )


# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""

#     pose_command = mdp.UniformPose2dCommandCfg(
#         asset_name="robot",
#         simple_heading=False,
#         resampling_time_range=(8.0, 8.0),
#         debug_vis=True,
#         ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
#     )


# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out = DoneTerm(func=mdp.time_out, time_out=True)
#     base_contact = DoneTerm(
#         func=mdp.illegal_contact,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
#     )


# @configclass
# class NavigationEnvCfg(ManagerBasedRLEnvCfg):
#     """Configuration for the navigation environment."""

#     # environment settings
#     scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
#     actions: ActionsCfg = ActionsCfg()
#     observations: ObservationsCfg = ObservationsCfg()
#     events: EventCfg = EventCfg()
#     # mdp settings
#     commands: CommandsCfg = CommandsCfg()
#     rewards: RewardsCfg = RewardsCfg()
#     terminations: TerminationsCfg = TerminationsCfg()

#     def __post_init__(self):
#         """Post initialization."""

#         self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
#         self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
#         self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
#         self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

#         if self.scene.height_scanner is not None:
#             self.scene.height_scanner.update_period = (
#                 self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
#             )
#         if self.scene.contact_forces is not None:
#             self.scene.contact_forces.update_period = self.sim.dt


# class NavigationEnvCfg_PLAY(NavigationEnvCfg):
#     def __post_init__(self) -> None:
#         # post init of parent
#         super().__post_init__()

#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False


## for waypoint navigation 

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


@configclass
class SequentialWaypointCommandCfg(CommandTermCfg):
    """Configuration for sequential waypoint command term."""
    
    asset_name: str = "robot"
    pattern: str = "star"
    num_waypoints: int = 5
    waypoint_radius: float = 0.5
    radius: float = 2.0
    debug_vis: bool = True
    
    def __post_init__(self):
        # Set the command class to SequentialWaypointCommand
        self.class_type = SequentialWaypointCommand
        super().__post_init__()


class SequentialWaypointCommand:
    """Command term that generates a sequence of waypoints for the robot to follow.
    
    When the robot reaches a waypoint, it advances to the next one in the sequence.
    All waypoints are visualized as cones in the environment.
    """
    
    def __init__(self, cfg, env):
        """Initialize the command term."""
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        
        # Generate waypoints
        self.waypoints = self._generate_waypoints()
        
        # Current waypoint indices
        self.current_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Prepare data tensor
        self.data = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Set up visualization for all waypoints
        if self.cfg.debug_vis:
            self._setup_visualization()
        
        self.update(env)
        
        print(f"[INFO] Initialized sequential waypoint command with {cfg.num_waypoints} waypoints")
        print(f"[INFO] Pattern: {cfg.pattern}, radius: {cfg.radius}, waypoint radius: {cfg.waypoint_radius}")
    
    def _setup_visualization(self):
        """Set up visualization markers for all waypoints."""
        try:
            # Import visualization modules
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RED_CONE_MARKER_CFG  # Use cones instead of spheres
            
            # Create markers for all waypoints (red cones)
            self.waypoint_visualizers = []
            for i in range(self.cfg.num_waypoints):
                marker_cfg = RED_CONE_MARKER_CFG.copy()
                marker_cfg.prim_path = f"/Visuals/Commands/waypoint_{i}"
                marker_cfg.markers["cone"].scale = (0.3, 0.3, 0.5)  # Larger cones
                self.waypoint_visualizers.append(VisualizationMarkers(marker_cfg))
            
            # Set visibility to true
            for visualizer in self.waypoint_visualizers:
                visualizer.set_visibility(True)
                
            print("[INFO] Successfully set up waypoint visualization")
        except Exception as e:
            print(f"[ERROR] Failed to set up visualization: {e}")
    
    def _update_visualization(self):
        """Update the visualization markers for all waypoints."""
        if not hasattr(self, 'target_visualizer'):
            return
            
        # Update current target visualization (green arrow)
        for i in range(self.num_envs):
            idx = self.current_indices[i].item()
            current_wp = self.waypoints[i, idx]
            
            # Position and orientation for current target
            pos = current_wp[:3].clone()
            pos[2] += 0.1  # Lift slightly above ground
            
            # Create quaternion from heading
            heading = current_wp[3]
            from isaaclab.utils import math as math_utils
            quat = math_utils.quat_from_euler_xyz(0.0, 0.0, heading)
            
            # Visualize current target
            self.target_visualizer.visualize(pos.unsqueeze(0), quat.unsqueeze(0))
        
        # Update all waypoint visualizations (blue spheres)
        for wp_idx in range(self.cfg.num_waypoints):
            positions = []
            for env_idx in range(self.num_envs):
                wp = self.waypoints[env_idx, wp_idx]
                pos = wp[:3].clone()
                pos[2] += 0.05  # Lift slightly above ground
                positions.append(pos)
            
            # Stack positions for batch visualization
            positions = torch.stack(positions, dim=0)
            
            # Identity quaternion for spheres
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            
            # Visualize waypoint
            self.waypoint_visualizers[wp_idx].visualize(positions, identity_quat)
    
    def _generate_waypoints(self):
        """Generate waypoints based on the configured pattern."""
        waypoints = torch.zeros((self.num_envs, self.cfg.num_waypoints, 4), device=self.device)
        
        for env_idx in range(self.num_envs):
            if self.cfg.pattern == "star":
                self._generate_star_pattern(waypoints, env_idx)
            elif self.cfg.pattern == "circle":
                self._generate_circle_pattern(waypoints, env_idx)
            elif self.cfg.pattern == "square":
                self._generate_square_pattern(waypoints, env_idx)
            else:
                # Default to star pattern
                self._generate_star_pattern(waypoints, env_idx)
        
        return waypoints
    
    def _generate_star_pattern(self, waypoints, env_idx):
        """Generate a star pattern of waypoints."""
        for wp_idx in range(self.cfg.num_waypoints):
            angle = 2 * math.pi * wp_idx / self.cfg.num_waypoints
            
            # Alternate between inner and outer points for star
            r = self.cfg.radius * (0.5 if wp_idx % 2 == 0 else 1.0)
            
            # Calculate position
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            # Calculate heading - point toward the next waypoint
            next_idx = (wp_idx + 1) % self.cfg.num_waypoints
            next_angle = 2 * math.pi * next_idx / self.cfg.num_waypoints
            next_r = self.cfg.radius * (0.5 if next_idx % 2 == 0 else 1.0)
            next_x = next_r * math.cos(next_angle)
            next_y = next_r * math.sin(next_angle)
            heading = math.atan2(next_y - y, next_x - x)
            
            # Store waypoint
            waypoints[env_idx, wp_idx] = torch.tensor([x, y, 0.0, heading], device=self.device)
    
    def _generate_circle_pattern(self, waypoints, env_idx):
        """Generate a circular pattern of waypoints."""
        for wp_idx in range(self.cfg.num_waypoints):
            angle = 2 * math.pi * wp_idx / self.cfg.num_waypoints
            
            # Calculate position on circle
            x = self.cfg.radius * math.cos(angle)
            y = self.cfg.radius * math.sin(angle)
            
            # Calculate heading - tangential to circle
            heading = angle + math.pi/2  # Tangent direction
            
            # Store waypoint
            waypoints[env_idx, wp_idx] = torch.tensor([x, y, 0.0, heading], device=self.device)
    
    def _generate_square_pattern(self, waypoints, env_idx):
        """Generate a square pattern of waypoints."""
        # Calculate points per side (minimum 1)
        points_per_side = max(1, self.cfg.num_waypoints // 4)
        
        # Generate points
        wp_idx = 0
        for side in range(4):  # 4 sides of square
            for i in range(points_per_side):
                if wp_idx >= self.cfg.num_waypoints:
                    break
                    
                # Calculate position
                t = (i + 0.5) / points_per_side  # Position along side (0.0 to 1.0)
                if side == 0:  # Top side (left to right)
                    x = -self.cfg.radius + 2 * self.cfg.radius * t
                    y = self.cfg.radius
                    heading = 0  # Facing right
                elif side == 1:  # Right side (top to bottom)
                    x = self.cfg.radius
                    y = self.cfg.radius - 2 * self.cfg.radius * t
                    heading = -math.pi/2  # Facing down
                elif side == 2:  # Bottom side (right to left)
                    x = self.cfg.radius - 2 * self.cfg.radius * t
                    y = -self.cfg.radius
                    heading = math.pi  # Facing left
                else:  # Left side (bottom to top)
                    x = -self.cfg.radius
                    y = -self.cfg.radius + 2 * self.cfg.radius * t
                    heading = math.pi/2  # Facing up
                
                # Store waypoint
                waypoints[env_idx, wp_idx] = torch.tensor([x, y, 0.0, heading], device=self.device)
                wp_idx += 1
        
        # Fill any remaining waypoints
        while wp_idx < self.cfg.num_waypoints:
            # Just duplicate the last point if we have extras
            waypoints[env_idx, wp_idx] = waypoints[env_idx, wp_idx-1]
            wp_idx += 1
    
    def reset(self, env):
        """Reset the command term."""
        # Reset current waypoint indices
        self.current_indices.zero_()
        
        # Update data with initial waypoints
        self.update(env)
        
        return self.data
    
    def update(self, env):
        """Update the command term."""
        # Get robot position
        robot = env.scene[self.cfg.asset_name]
        robot_pos = robot.data.root_pos_w
        
        # Check if waypoints are reached
        for i in range(self.num_envs):
            idx = self.current_indices[i].item()
            current_wp = self.waypoints[i, idx, :2]
            
            # Calculate distance to waypoint
            distance = torch.norm(robot_pos[i, :2] - current_wp)
            
            # If waypoint is reached, move to next one
            if distance < self.cfg.waypoint_radius:
                next_idx = (idx + 1) % self.cfg.num_waypoints
                self.current_indices[i] = next_idx
                print(f"[ENV {i}] Reached waypoint {idx}, moving to waypoint {next_idx}")
        
        # Update data with current waypoints
        for i in range(self.num_envs):
            idx = self.current_indices[i].item()
            self.data[i] = self.waypoints[i, idx]
        
        # Update visualization
        if self.cfg.debug_vis and hasattr(self, 'target_visualizer'):
            self._update_visualization()
        
        return self.data


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )


@configclass
class WaypointCommandsCfg:
    """Command terms for waypoint navigation."""

    pose_command = SequentialWaypointCommandCfg(
        asset_name="robot",
        pattern="star",
        num_waypoints=5,
        waypoint_radius=0.5,
        radius=2.0,
        debug_vis=True,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


class WaypointNavigationEnvCfg(NavigationEnvCfg):
    """Configuration for navigation with waypoints."""
    
    # Override commands to use waypoint commands
    commands: WaypointCommandsCfg = WaypointCommandsCfg()
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # Set a longer episode length for waypoint following
        self.episode_length_s = 60.0


class WaypointNavigationEnvCfg_PLAY(WaypointNavigationEnvCfg):
    """Configuration for waypoint navigation in play mode."""
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 10  # Fewer environments for clearer visualization
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False