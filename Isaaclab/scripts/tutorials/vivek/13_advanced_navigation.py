#!/usr/bin/env python3

"""
Tutorial 13: Advanced Navigation to Coordinates
===============================================

This tutorial demonstrates how to create a manager-based environment 
for a legged robot to navigate to specific coordinates. We build on the
locomotion tutorial but add these advanced features:

1. Goal-directed navigation to specific world coordinates
2. Velocity-based control for precise movement
3. Adaptive navigation with path correction
4. Visualization of navigation targets

This provides a foundation for more complex autonomous navigation tasks.
"""

import math
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Union, Any

# IsaacLab imports
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.managers import SceneEntityManager, ObservationManager
from omni.isaac.lab.managers import RewardManager, ResetManager, TerminationManager
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils.dict_utils import to_torch, deep_dict_clone
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import quat_rotate_inverse, quat_conjugate
from omni.isaac.lab.envs import VecEnvMT
from omni.isaac.lab.utils.torch import prepare_tensor

# Visualization
from omni.isaac.lab.utils.timer import Timer
from omni.isaac.lab.sim import SimulationContext
import omni.kit.commands
from pxr import UsdGeom, Gf


def main():
    """Main function for the navigation tutorial."""
    # Launch the simulation
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    # Create a simulation context
    sim_context = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params={"physx": {"solver_type": 1, "num_position_iterations": 4, "num_velocity_iterations": 1}}
    )

    # Create the interactive scene
    scene = InteractiveScene(sim_context)
    scene.add_default_ground_plane()
    
    # Create the navigation environment
    env = NavigationEnv(scene)
    
    # Run the simulation
    sim_context.reset()
    env.reset()
    
    # Print instructions
    print("=" * 80)
    print("Navigation Environment Tutorial")
    print("=" * 80)
    print("This demo shows a quadruped robot navigating to target locations.")
    print("The robot will try to reach randomly placed targets in the environment.")
    print("Green sphere indicates the current target location.")
    
    # Create visualization for target
    target_sphere = create_target_indicator(scene)
    
    # Run simulation with auto-navigation controller
    timer = Timer()
    timer.start()
    duration = 60.0  # Run for 60 seconds
    
    while timer.time() < duration:
        # Get observations
        obs_dict = env._observation_manager.compute_observations()
        goal_pos_local = obs_dict["goal_pos_local"][0]  # Local coordinates of goal
        
        # Update target visualization
        goal_pos_world = obs_dict["goal_pos_world"][0]
        update_target_indicator(target_sphere, goal_pos_world)
        
        # Simple navigation controller: convert local goal position to velocity commands
        actions = compute_navigation_actions(goal_pos_local)
        actions = to_torch(actions, device="cpu", dtype=torch.float32)
        
        # Step the environment
        obs, rewards, dones, infos = env.step(actions)
        
        # Reset if episode is done
        if dones.any():
            env.reset()
            
        # Step the simulation
        sim_context.step()
        
        # Sleep to maintain real-time
        time.sleep(0.01)
    
    # Clean up
    env.close()
    simulation_app.close()


def create_target_indicator(scene: InteractiveScene) -> str:
    """Create a visual indicator for the target location.
    
    Args:
        scene: The interactive scene to add the indicator to.
        
    Returns:
        The path to the created sphere.
    """
    # Create a sphere to indicate the target
    sphere_path = "/World/target_indicator"
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Sphere",
        prim_path=sphere_path,
        attributes={"radius": 0.3}
    )
    
    # Set the color to green
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Material",
        prim_path=f"{sphere_path}/Material",
        attributes={}
    )
    
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Shader",
        prim_path=f"{sphere_path}/Material/Shader",
        attributes={"info:id": "UsdPreviewSurface"}
    )
    
    omni.kit.commands.execute(
        "SetAttributeCommand",
        attr_path=f"{sphere_path}/Material/Shader.inputs:diffuseColor",
        value=Gf.Vec3f(0.0, 1.0, 0.0)  # Green color
    )
    
    omni.kit.commands.execute(
        "ConnectAttribute",
        source_path=f"{sphere_path}/Material/Shader.outputs:surface",
        target_path=f"{sphere_path}/Material.outputs:surface"
    )
    
    omni.kit.commands.execute(
        "BindMaterial",
        material_path=f"{sphere_path}/Material",
        prim_path=sphere_path
    )
    
    return sphere_path


def update_target_indicator(sphere_path: str, position: torch.Tensor) -> None:
    """Update the position of the target indicator.
    
    Args:
        sphere_path: The path to the sphere prim.
        position: The position to move the sphere to.
    """
    # Convert position tensor to Gf.Vec3d
    pos = Gf.Vec3d(position[0].item(), position[1].item(), position[2].item())
    
    # Set the position of the sphere
    omni.kit.commands.execute(
        "TransformPrimCommand",
        path=sphere_path,
        translation=pos
    )


def compute_navigation_actions(goal_pos_local: torch.Tensor) -> np.ndarray:
    """Compute actions to navigate to the goal position.
    
    This is a simple proportional controller that generates velocity
    commands based on the direction and distance to the goal.
    
    Args:
        goal_pos_local: The goal position in the robot's local frame.
        
    Returns:
        An array of actions for the robot.
    """
    # Extract local goal coordinates
    goal_x = goal_pos_local[0].item()
    goal_y = goal_pos_local[1].item()
    goal_dist = np.sqrt(goal_x**2 + goal_y**2)
    
    # Compute desired velocity command (forward and lateral)
    # Scale based on distance to goal
    max_vel = 1.0
    desired_vel_x = np.clip(goal_x, -max_vel, max_vel)
    desired_vel_y = np.clip(goal_y, -max_vel, max_vel)
    
    # Compute desired yaw command (to face the goal)
    desired_yaw = np.arctan2(goal_y, goal_x)
    # Scale yaw command
    desired_yaw = np.clip(desired_yaw, -0.5, 0.5)
    
    # Construct command [vx, vy, yaw]
    vel_command = np.array([desired_vel_x, desired_vel_y, desired_yaw])
    
    # When close to goal, reduce velocity
    slowdown_distance = 1.0
    if goal_dist < slowdown_distance:
        vel_command[0:2] *= goal_dist / slowdown_distance
    
    # Our action is just the velocity command (will be mapped to joint actions by the env)
    return vel_command.reshape(1, 3)


class NavigationEnv(VecEnvMT):
    """A manager-based environment for quadruped navigation tasks."""
    
    def __init__(self, scene: InteractiveScene):
        """Initialize the navigation environment.
        
        Args:
            scene: The interactive scene to use.
        """
        # Store the scene
        self._scene = scene
        
        # Create the managers
        self._scene_manager = SceneEntityManager(
            scene=self._scene,
            cfg=self._get_scene_entity_cfg(),
        )
        
        self._observation_manager = NavigationObservationManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._reward_manager = NavigationRewardManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._reset_manager = NavigationResetManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._termination_manager = NavigationTerminationManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        # Initialize the base class
        super().__init__(
            num_envs=1,
            scene=self._scene,
            scene_entity_manager=self._scene_manager,
            observation_manager=self._observation_manager,
            reset_manager=self._reset_manager,
            termination_manager=self._termination_manager,
            reward_manager=self._reward_manager,
        )
        
    def _get_scene_entity_cfg(self) -> Dict[str, SceneEntityCfg]:
        """Create the scene entity configuration for the navigation environment.
        
        Returns:
            A dictionary of scene entity configurations.
        """
        # Configure a quadruped robot
        cfg = {
            "robot": SceneEntityCfg(
                # Use the ANYmal C robot from Isaac Lab
                "omni.isaac.lab.assets.anymal_c.AnymalC",
                # Robot spawn pose
                spawn={"pos": (0, 0, 0.5), "rot": (0, 0, 0)},
                # Initial joint positions and velocities
                init_state={
                    "joint_pos": {
                        # Initial joint positions for a standing pose
                        "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
                        "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
                        "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
                        "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
                    },
                    "joint_vel": {
                        # Zero initial velocities
                        "LF_HAA": 0.0, "LF_HFE": 0.0, "LF_KFE": 0.0,
                        "RF_HAA": 0.0, "RF_HFE": 0.0, "RF_KFE": 0.0,
                        "LH_HAA": 0.0, "LH_HFE": 0.0, "LH_KFE": 0.0,
                        "RH_HAA": 0.0, "RH_HFE": 0.0, "RH_KFE": 0.0,
                    },
                },
                # Keep robot articulation fixed (not affected by gravity) during reset
                rigid_props={"disable_gravity": False},
            ),
        }
        return cfg
    
    @property
    def action_dim(self) -> int:
        """Return the dimension of the action space."""
        # 3 for [forward_velocity, lateral_velocity, yaw_rate]
        return 3


class NavigationObservationManager(ObservationManager):
    """Observation manager for navigation tasks."""
    
    def __init__(self, scene: InteractiveScene, scene_manager: SceneEntityManager):
        """Initialize the observation manager.
        
        Args:
            scene: The interactive scene.
            scene_manager: The scene entity manager.
        """
        super().__init__(scene, scene_manager)
        
        # Get all the joint names for the robot
        self.robot = scene_manager.entity_map["robot"]
        self.joint_names = self.robot.dof_names
        self.num_joints = len(self.joint_names)
        
        # Navigation goal parameters
        self.goal_position = torch.tensor([[5.0, 0.0, 0.0]])  # Default goal
        
        # Define observation groups
        self._observation_groups = {
            "policy": [
                "joint_pos",        # Current joint positions
                "joint_vel",        # Current joint velocities
                "base_lin_vel",     # Base linear velocity
                "base_ang_vel",     # Base angular velocity
                "projected_gravity",# Gravity vector in robot frame
                "goal_pos_local",   # Goal position in robot local frame
                "heading_to_goal",  # Heading angle to goal
                "dist_to_goal",     # Distance to goal
                "feet_contact",     # Binary indicators for feet contact
            ]
        }
        
    def compute_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for the current state.
        
        Returns:
            A dictionary of observation tensors.
        """
        observations = {}
        
        # Get joint states
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        
        # Get base states
        base_pos, base_quat = self.robot.get_world_pose(body_name="base")
        base_lin_vel, base_ang_vel = self.robot.get_body_velocity(body_name="base")
        
        # Project gravity into the robot's local frame
        gravity = torch.tensor([0.0, 0.0, -9.81], device=base_pos.device).repeat(1, 1)
        base_rot_mat = self.robot.get_world_pose_rotmat(body_name="base")[1]
        projected_gravity = torch.bmm(base_rot_mat.transpose(1, 2), gravity.unsqueeze(-1)).squeeze(-1)
        
        # Ensure goal position is on the same device as base_pos
        goal_pos_world = self.goal_position.to(base_pos.device)
        
        # Compute goal position in robot's local frame
        goal_pos_relative = goal_pos_world - base_pos
        goal_pos_local = quat_rotate_inverse(base_quat, goal_pos_relative)
        
        # Compute distance to goal
        dist_to_goal = torch.norm(goal_pos_relative[:, :2], dim=1, keepdim=True)  # Only x-y distance
        
        # Compute heading to goal (angle in the x-y plane)
        heading_to_goal = torch.atan2(goal_pos_local[:, 1], goal_pos_local[:, 0]).unsqueeze(1)
        
        # Feet contact states
        feet_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        feet_contact = torch.zeros((1, len(feet_names)), device=base_pos.device)
        
        # For this tutorial, we'll use a simplified check based on height
        for i, foot_name in enumerate(feet_names):
            foot_pos, _ = self.robot.get_world_pose(body_name=foot_name)
            # Check if foot is close to the ground
            feet_contact[0, i] = 1.0 if foot_pos[0, 2] < 0.05 else 0.0
        
        # Store all observations
        observations["joint_pos"] = joint_pos
        observations["joint_vel"] = joint_vel
        observations["base_lin_vel"] = base_lin_vel
        observations["base_ang_vel"] = base_ang_vel
        observations["projected_gravity"] = projected_gravity
        observations["goal_pos_world"] = goal_pos_world  # For visualization
        observations["goal_pos_local"] = goal_pos_local
        observations["heading_to_goal"] = heading_to_goal
        observations["dist_to_goal"] = dist_to_goal
        observations["feet_contact"] = feet_contact
        
        return observations
    
    def process_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process velocity commands and convert to joint position targets.
        
        Args:
            actions: The velocity commands [forward_vel, lateral_vel, yaw_rate].
            
        Returns:
            A dictionary of processed actions.
        """
        # Extract velocity commands
        forward_vel = actions[:, 0:1]  # Forward velocity
        lateral_vel = actions[:, 1:2]  # Lateral velocity
        yaw_rate = actions[:, 2:3]     # Turning rate
        
        # Convert to default joint positions + leg movement patterns
        default_joint_positions = torch.tensor([
            # Left Front leg
            0.0, 0.4, -0.8,
            # Right Front leg
            0.0, 0.4, -0.8,
            # Left Hind leg
            0.0, -0.4, 0.8,
            # Right Hind leg
            0.0, -0.4, 0.8
        ], device=actions.device).unsqueeze(0)
        
        # Simplified leg movement model based on velocity
        # In a real implementation, you'd use a proper gait generator
        # This is just for demonstration
        
        # Scale leg motion based on velocity magnitude
        vel_magnitude = torch.sqrt(forward_vel**2 + lateral_vel**2)
        
        # Generate a simple trotting pattern
        # (in real implementation, you would use a proper gait generator)
        phase = time.time() * 5.0  # Simple time-based phase
        
        # Modulate the hip and knee joints based on phase and velocity
        leg_pattern = torch.zeros((1, 12), device=actions.device)
        
        # Simple trot gait (opposite legs move together)
        # Left front and right hind move together
        # Right front and left hind move together
        lf_rh_phase = torch.sin(torch.tensor([phase]))
        rf_lh_phase = torch.sin(torch.tensor([phase + math.pi]))
        
        # Apply to hip and knee joints with scaling based on velocity
        stride_scale = 0.2 * vel_magnitude
        
        # Left front leg
        leg_pattern[0, 1] = lf_rh_phase * stride_scale  # Hip
        leg_pattern[0, 2] = -lf_rh_phase * stride_scale * 1.5  # Knee (larger motion)
        
        # Right front leg
        leg_pattern[0, 4] = rf_lh_phase * stride_scale
        leg_pattern[0, 5] = -rf_lh_phase * stride_scale * 1.5
        
        # Left hind leg
        leg_pattern[0, 7] = rf_lh_phase * stride_scale
        leg_pattern[0, 8] = rf_lh_phase * stride_scale * 1.5
        
        # Right hind leg
        leg_pattern[0, 10] = lf_rh_phase * stride_scale
        leg_pattern[0, 11] = lf_rh_phase * stride_scale * 1.5
        
        # Apply turning motion to abduction joints
        turn_scale = 0.2
        
        # When turning, abduct legs on one side more than the other
        # Left legs (0, 6)
        leg_pattern[0, 0] = yaw_rate * turn_scale  # Left front abduction
        leg_pattern[0, 6] = yaw_rate * turn_scale  # Left hind abduction
        
        # Right legs (3, 9)
        leg_pattern[0, 3] = -yaw_rate * turn_scale  # Right front abduction
        leg_pattern[0, 9] = -yaw_rate * turn_scale  # Right hind abduction
        
        # Calculate final joint positions
        joint_pos_target = default_joint_positions + leg_pattern.to(actions.device)
        
        # Apply target positions
        self.robot.set_joint_position_targets(joint_pos_target.squeeze(0))
        
        return {"joint_pos_target": joint_pos_target}


class NavigationRewardManager(RewardManager):
    """Reward manager for navigation tasks."""
    
    def __init__(self, scene: InteractiveScene, scene_manager: SceneEntityManager):
        """Initialize the reward manager.
        
        Args:
            scene: The interactive scene.
            scene_manager: The scene entity manager.
        """
        super().__init__(scene, scene_manager)
        
        # Get robot from scene manager
        self.robot = scene_manager.entity_map["robot"]
        
        # Define reward terms and their weights
        self.reward_scales = {
            "tracking_lin_vel": 0.5,      # Reward for maintaining commanded velocity  
            "goal_progress": 1.0,         # Reward for approaching the goal
            "goal_reached": 10.0,         # Bonus for reaching the goal
            "heading_alignment": 0.2,     # Reward for facing toward the goal
            "joint_power": -0.0001,       # Penalty for joint power/energy use
            "orientation": -0.1,          # Penalty for non-upright orientation
            "feet_contact": 0.05          # Small reward for making ground contact with feet
        }
        
        # Store previous values for computing deltas
        self._prev_goal_distance = None
        
    def compute_rewards(self) -> Dict[str, torch.Tensor]:
        """Compute rewards for the current state.
        
        Returns:
            A dictionary of reward tensors.
        """
        rewards = {}
        
        # Get relevant states
        base_lin_vel, base_ang_vel = self.robot.get_body_velocity(body_name="base")
        _, base_quat = self.robot.get_world_pose(body_name="base")
        
        # Get observation manager for navigation-specific data
        obs_manager = self._scene.env_mgmt._observation_manager
        observations = obs_manager.compute_observations()
        goal_pos_local = observations["goal_pos_local"]
        dist_to_goal = observations["dist_to_goal"]
        heading_to_goal = observations["heading_to_goal"]
        feet_contact = observations["feet_contact"]
        
        # 1. Reward for progress toward the goal
        if self._prev_goal_distance is not None:
            # Reward is positive when getting closer to the goal
            goal_progress = self._prev_goal_distance - dist_to_goal
            rewards["goal_progress"] = goal_progress * self.reward_scales["goal_progress"]
        else:
            rewards["goal_progress"] = torch.zeros(1, dtype=torch.float32, device=dist_to_goal.device)
        
        # Store current distance for next frame
        self._prev_goal_distance = dist_to_goal.clone()
        
        # 2. Bonus for reaching the goal
        goal_reached = (dist_to_goal < 0.5).float()  # Within half a meter
        rewards["goal_reached"] = goal_reached * self.reward_scales["goal_reached"]
        
        # 3. Heading alignment reward
        # Reward is maximized when heading directly toward the goal
        heading_alignment = torch.cos(heading_to_goal)
        rewards["heading_alignment"] = heading_alignment * self.reward_scales["heading_alignment"]
        
        # 4. Linear velocity tracking reward
        # Ideal velocity is proportional to distance and direction to goal
        # We're simplifying here - a more robust implementation would 
        # compute ideal velocity based on a path planner
        
        ideal_forward_vel = goal_pos_local[:, 0].clamp(-1.0, 1.0)
        ideal_lateral_vel = goal_pos_local[:, 1].clamp(-1.0, 1.0)
        
        # When close to goal, reduce ideal velocity
        slowdown_dist = 1.0
        close_to_goal = (dist_to_goal < slowdown_dist).float()
        ideal_forward_vel = ideal_forward_vel * (1 - close_to_goal) + ideal_forward_vel * dist_to_goal / slowdown_dist * close_to_goal
        ideal_lateral_vel = ideal_lateral_vel * (1 - close_to_goal) + ideal_lateral_vel * dist_to_goal / slowdown_dist * close_to_goal
        
        # Compute error between current and ideal velocity
        vel_error = torch.sqrt(
            (base_lin_vel[:, 0] - ideal_forward_vel)**2 + 
            (base_lin_vel[:, 1] - ideal_lateral_vel)**2
        ).unsqueeze(1)
        
        # Reward is higher when error is lower (exponential decay)
        rewards["tracking_lin_vel"] = torch.exp(-vel_error / 0.25) * self.reward_scales["tracking_lin_vel"]
        
        # 5. Joint power penalty (energy efficiency)
        joint_efforts = self.robot.get_applied_joint_efforts()
        joint_vel = self.robot.get_joint_velocities()
        joint_power = torch.sum(torch.abs(joint_vel * joint_efforts), dim=1)
        rewards["joint_power"] = joint_power * self.reward_scales["joint_power"]
        
        # 6. Orientation reward/penalty
        # We want the z-axis to point up (0, 0, 1) in world frame
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=base_quat.device).repeat(1, 1)
        up_vector_local = self.robot.get_world_pose_rotmat(body_name="base")[1].transpose(1, 2) @ up_vector.unsqueeze(-1)
        up_vector_local = up_vector_local.squeeze(-1)
        
        # Dot product with local z-axis [0, 0, 1]
        alignment = up_vector_local[:, 2]  # z-component
        orientation_error = 1.0 - alignment
        rewards["orientation"] = orientation_error * self.reward_scales["orientation"]
        
        # 7. Foot contact reward (small reward for making contact)
        contact_reward = torch.sum(feet_contact, dim=1)
        rewards["feet_contact"] = contact_reward * self.reward_scales["feet_contact"]
        
        # Total reward is sum of all reward terms
        rewards["total"] = sum(rewards.values())
        
        return rewards


class NavigationResetManager(ResetManager):
    """Reset manager for navigation tasks."""
    
    def __init__(self, scene: InteractiveScene, scene_manager: SceneEntityManager):
        """Initialize the reset manager.
        
        Args:
            scene: The interactive scene.
            scene_manager: The scene entity manager.
        """
        super().__init__(scene, scene_manager)
        
        # Get robot from scene manager
        self.robot = scene_manager.entity_map["robot"]
        
        # Reset noise parameters (used to add randomness to initial conditions)
        self.pos_noise = 0.1  # Position noise in meters
        self.rot_noise = 0.1  # Rotation noise in radians
        
        # Navigation parameters
        self.min_goal_dist = 3.0   # Minimum distance to goal
        self.max_goal_dist = 8.0   # Maximum distance to goal
        self.goal_z = 0.0          # Height of the goal
        
    def reset(self) -> None:
        """Reset the environment to initial conditions with randomization."""
        # Default joint positions for standing
        default_joint_positions = {
            # Left Front leg
            "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
            # Right Front leg
            "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
            # Left Hind leg
            "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
            # Right Hind leg
            "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
        }
        
        # Add noise to joint positions
        joint_positions = {}
        for name, pos in default_joint_positions.items():
            joint_positions[name] = pos + np.random.uniform(-0.05, 0.05)
        
        # Reset joint positions
        self.robot.set_joint_positions(joint_positions)
        
        # Reset joint velocities to zero
        zero_vels = {name: 0.0 for name in self.robot.dof_names}
        self.robot.set_joint_velocities(zero_vels)
        
        # Reset base position with small random offset
        x_noise = np.random.uniform(-self.pos_noise, self.pos_noise)
        y_noise = np.random.uniform(-self.pos_noise, self.pos_noise)
        
        # Set base pose
        self.robot.set_world_pose(
            position=(x_noise, y_noise, 0.5),
            orientation=(0, 0, np.random.uniform(-self.rot_noise, self.rot_noise))
        )
        
        # Reset base velocity to zero
        self.robot.set_body_velocity(
            linear_velocity=(0, 0, 0),
            angular_velocity=(0, 0, 0),
            body_name="base"
        )
        
        # Generate a new random goal position
        self._randomize_goal_position()
        
        # Reset progress tracking
        obs_manager = self._scene.env_mgmt._observation_manager
        reward_manager = self._scene.env_mgmt._reward_manager
        
        # In reward manager, reset the previous distance
        reward_manager._prev_goal_distance = None
        
    def _randomize_goal_position(self) -> None:
        """Generate a random goal position within specified bounds."""
        # Get the observation manager
        obs_manager = self._scene.env_mgmt._observation_manager
        
        # Get robot's current position
        robot_pos, _ = self.robot.get_world_pose(body_name="base")
        
        # Generate random distance and angle
        distance = np.random.uniform(self.min_goal_dist, self.max_goal_dist)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Calculate goal position in world frame
        goal_x = robot_pos[0, 0].item() + distance * np.cos(angle)
        goal_y = robot_pos[0, 1].item() + distance * np.sin(angle)
        
        # Set the goal position (at ground level)
        goal_position = torch.tensor([[goal_x, goal_y, self.goal_z]])
        obs_manager.goal_position = goal_position


class NavigationTerminationManager(TerminationManager):
    """Termination manager for navigation tasks."""
    
    def __init__(self, scene: InteractiveScene, scene_manager: SceneEntityManager):
        """Initialize the termination manager.
        
        Args:
            scene: The interactive scene.
            scene_manager: The scene entity manager.
        """
        super().__init__(scene, scene_manager)
        
        # Get robot from scene manager
        self.robot = scene_manager.entity_map["robot"]
        
        # Define termination thresholds
        self.max_episode_length = 20.0  # Maximum episode length in seconds
        self.goal_threshold = 0.5       # Distance threshold to consider goal reached
        self.max_height = 1.0           # Maximum height of the base above ground
        self.min_height = 0.2           # Minimum height of the base above ground
        self.max_roll = 1.0             # Maximum roll angle in radians
        self.max_pitch = 1.0            # Maximum pitch angle in radians
        
        # Track episode time
        self._episode_time = 0.0
        
    def reset(self) -> None:
        """Reset the termination manager."""
        self._episode_time = 0.0
        
    def get_terminations(self) -> Dict[str, torch.Tensor]:
        """Get termination signals for the current state.
        
        Returns:
            A dictionary of termination signals.
        """
        terminations = {}
        device = self.robot.device
        
        # Update episode time
        self._episode_time += self._scene.physics_sim_view.get_physics_dt()
        
        # Get base pose
        pos, quat = self.robot.get_world_pose(body_name="base")
        
        # Get goal information
        obs_manager = self._scene.env_mgmt._observation_manager
        observations = obs_manager.compute_observations()
        dist_to_goal = observations["dist_to_goal"]
        
        # Convert quaternion to Euler angles
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        roll, pitch, yaw = quat_to_euler_angles(quat)
        
        # Goal reached termination
        goal_reached = (dist_to_goal < self.goal_threshold)
        
        # Time limit termination
        time_limit = torch.tensor([self._episode_time > self.max_episode_length], 
                                   dtype=torch.bool, device=device)
        
        # Height termination
        height_violation = torch.logical_or(
            pos[:, 2] > self.max_height,
            pos[:, 2] < self.min_height
        )
        
        # Orientation termination
        roll_violation = torch.abs(roll) > self.max_roll
        pitch_violation = torch.abs(pitch) > self.max_pitch
        orientation_violation = torch.logical_or(roll_violation, pitch_violation)
        
        # Combine termination conditions
        base_termination = torch.logical_or(height_violation, orientation_violation)
        
        # Store termination signals
        terminations["goal_reached"] = goal_reached
        terminations["time_limit"] = time_limit
        terminations["base_termination"] = base_termination
        
        # Combined termination signal
        terminations["terminated"] = torch.logical_or(
            torch.logical_or(goal_reached, time_limit),
            base_termination
        )
        
        return terminations


if __name__ == "__main__":
    main() 