#!/usr/bin/env python3

"""
Tutorial 12: Manager-Based Locomotion Environment
=================================================

This tutorial demonstrates how to create a manager-based environment
for quadruped locomotion tasks. We'll build on the concepts from
the previous tutorial, but focus specifically on the components
needed for locomotion tasks including:

1. Scene setup with a quadruped robot
2. Specialized locomotion observations and actions
3. Locomotion-specific reward functions
4. Reset and termination conditions for locomotion

This provides a foundation for training policies for legged robots.
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
from omni.isaac.lab.envs import VecEnvMT

# Visualization
from omni.isaac.lab.utils.timer import Timer
from omni.isaac.lab.sim import SimulationContext

# For visualization
import omni.kit.commands


def main():
    """Main function for the locomotion tutorial."""
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
    
    # Create the locomotion environment
    env = LocomotionEnv(scene)
    
    # Run the simulation
    sim_context.reset()
    env.reset()
    
    # Print instructions
    print("=" * 80)
    print("Locomotion Environment Tutorial")
    print("=" * 80)
    print("This demo shows a quadruped robot learning to walk.")
    print("We'll use random actions to demonstrate the environment.")
    
    # Run some iterations with random actions
    timer = Timer()
    timer.start()
    duration = 30.0  # Run for 30 seconds
    
    while timer.time() < duration:
        # Step with random actions
        actions = np.random.uniform(-1.0, 1.0, size=(1, env.action_dim))
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
    

class LocomotionEnv(VecEnvMT):
    """A manager-based environment for quadruped locomotion tasks."""
    
    def __init__(self, scene: InteractiveScene):
        """Initialize the locomotion environment.
        
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
        
        self._observation_manager = LocomotionObservationManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._reward_manager = LocomotionRewardManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._reset_manager = LocomotionResetManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._termination_manager = LocomotionTerminationManager(
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
        """Create the scene entity configuration for the locomotion environment.
        
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
        # 12 joints for a quadruped (3 per leg x 4 legs)
        return 12


class LocomotionObservationManager(ObservationManager):
    """Observation manager for locomotion tasks."""
    
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
        
        # Define observation groups
        self._observation_groups = {
            "policy": [
                "joint_pos",        # Current joint positions
                "joint_vel",        # Current joint velocities
                "joint_pos_target", # Target joint positions (from previous action)
                "base_lin_vel",     # Base linear velocity
                "base_ang_vel",     # Base angular velocity
                "projected_gravity",# Gravity vector in robot frame
                "command",          # Command velocity
                "feet_contact",     # Binary indicators for feet contact
            ]
        }
                
        # Default command velocity (forward walking)
        self.command_velocity = torch.tensor([[1.0, 0.0, 0.0]])
        
    def compute_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for the current state.
        
        Returns:
            A dictionary of observation tensors.
        """
        observations = {}
        
        # Get joint states
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        
        # Get previous action as joint position targets
        if hasattr(self, "_joint_pos_target"):
            joint_pos_target = self._joint_pos_target
        else:
            # First step, use current positions
            joint_pos_target = joint_pos.clone()
            self._joint_pos_target = joint_pos_target
        
        # Get base states
        base_pos, base_quat = self.robot.get_world_pose(body_name="base")
        base_lin_vel, base_ang_vel = self.robot.get_body_velocity(body_name="base")
        
        # Project gravity into the robot's local frame
        gravity = torch.tensor([0.0, 0.0, -9.81], device=base_pos.device).repeat(1, 1)
        base_rot_mat = self.robot.get_world_pose_rotmat(body_name="base")[1]
        projected_gravity = torch.bmm(base_rot_mat.transpose(1, 2), gravity.unsqueeze(-1)).squeeze(-1)
        
        # Feet contact states
        feet_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        feet_contact = torch.zeros((1, len(feet_names)), device=base_pos.device)
        
        # TODO: In a real implementation, we'd compute actual contact forces
        # For this tutorial, we'll use a simplified check based on height
        for i, foot_name in enumerate(feet_names):
            foot_pos, _ = self.robot.get_world_pose(body_name=foot_name)
            # Check if foot is close to the ground
            feet_contact[0, i] = 1.0 if foot_pos[0, 2] < 0.05 else 0.0
                
        # Store observations
        observations["joint_pos"] = joint_pos
        observations["joint_vel"] = joint_vel
        observations["joint_pos_target"] = joint_pos_target
        observations["base_lin_vel"] = base_lin_vel
        observations["base_ang_vel"] = base_ang_vel
        observations["projected_gravity"] = projected_gravity
        observations["command"] = self.command_velocity.to(base_pos.device)
        observations["feet_contact"] = feet_contact
        
        return observations
    
    def process_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process actions to control the robot.
        
        Args:
            actions: The actions to apply.
            
        Returns:
            A dictionary of processed actions.
        """
        # Scale actions from [-1, 1] to joint limits
        # For this tutorial, we'll use a simple PD controller approach
        
        # Get default joint positions for standing
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
        
        # Scale actions to joint angle targets
        # Allow for +/- 0.5 radians from default pose
        joint_pos_target = default_joint_positions + actions * 0.5
        
        # Store the target for the next observation
        self._joint_pos_target = joint_pos_target
        
        # Apply PD control to reach target positions
        # Apply target positions
        self.robot.set_joint_position_targets(joint_pos_target.squeeze(0))
        
        return {"joint_pos_target": joint_pos_target}


class LocomotionRewardManager(RewardManager):
    """Reward manager for locomotion tasks."""
    
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
            "tracking_lin_vel": 1.0,      # Reward for tracking linear velocity command
            "tracking_ang_vel": 0.5,      # Reward for tracking angular velocity command
            "joint_accel": -0.01,         # Penalty for joint accelerations (smoothness)
            "action_rate": -0.01,         # Penalty for changing actions too quickly
            "joint_power": -0.0005,       # Penalty for joint power/energy use
            "orientation": -0.1,          # Penalty for non-upright orientation
            "feet_air_time": 0.2,         # Reward for feet spending time in the air (encourages walking)
            "feet_contact": 0.05          # Small reward for making ground contact with feet
        }
        
        # Store previous values for computing deltas
        self._prev_joint_vel = None
        self._prev_actions = None
        
        # Track feet air time
        self._feet_air_time = torch.zeros((1, 4), dtype=torch.float32)
        
    def compute_rewards(self) -> Dict[str, torch.Tensor]:
        """Compute rewards for the current state.
        
        Returns:
            A dictionary of reward tensors.
        """
        rewards = {}
        
        # Get relevant states
        base_lin_vel, base_ang_vel = self.robot.get_body_velocity(body_name="base")
        _, base_quat = self.robot.get_world_pose(body_name="base")
        joint_vel = self.robot.get_joint_velocities()
        
        # Get observation manager for its data
        obs_manager = self._scene.env_mgmt._observation_manager
        command = obs_manager.command_velocity
        feet_contact = obs_manager.compute_observations()["feet_contact"]
        
        # 1. Velocity tracking reward
        # Calculate how well the robot is following the commanded velocity
        lin_vel_error = torch.sum(torch.square(base_lin_vel - command), dim=1)
        rewards["tracking_lin_vel"] = torch.exp(-lin_vel_error / 0.25) * self.reward_scales["tracking_lin_vel"]
        
        # No angular velocity command for now, so reward staying upright
        target_ang_vel = torch.zeros_like(base_ang_vel)
        ang_vel_error = torch.sum(torch.square(base_ang_vel - target_ang_vel), dim=1)
        rewards["tracking_ang_vel"] = torch.exp(-ang_vel_error / 0.25) * self.reward_scales["tracking_ang_vel"]
        
        # 2. Joint acceleration penalty
        if self._prev_joint_vel is not None:
            joint_accel = (joint_vel - self._prev_joint_vel) / self._scene.physics_sim_view.get_physics_dt()
            joint_accel_penalty = torch.sum(torch.square(joint_accel), dim=1)
            rewards["joint_accel"] = joint_accel_penalty * self.reward_scales["joint_accel"]
        else:
            rewards["joint_accel"] = torch.zeros(1, dtype=torch.float32, device=base_lin_vel.device)
        
        # Store joint velocities for next frame
        self._prev_joint_vel = joint_vel.clone()
        
        # 3. Action rate penalty (if changing actions too quickly)
        if hasattr(self, "_prev_actions") and self._prev_actions is not None:
            action_rate_penalty = torch.sum(torch.square(
                obs_manager._joint_pos_target - self._prev_actions), dim=1)
            rewards["action_rate"] = action_rate_penalty * self.reward_scales["action_rate"]
        else:
            rewards["action_rate"] = torch.zeros(1, dtype=torch.float32, device=base_lin_vel.device)
        
        # Store current actions for next frame
        self._prev_actions = obs_manager._joint_pos_target.clone()
        
        # 4. Joint power penalty (energy efficiency)
        joint_power = torch.sum(torch.abs(joint_vel * self.robot.get_applied_joint_efforts()), dim=1)
        rewards["joint_power"] = joint_power * self.reward_scales["joint_power"]
        
        # 5. Orientation reward/penalty
        # We want the z-axis to point up (0, 0, 1) in world frame
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=base_quat.device).repeat(1, 1)
        up_vector_local = self.robot.get_world_pose_rotmat(body_name="base")[1].transpose(1, 2) @ up_vector.unsqueeze(-1)
        up_vector_local = up_vector_local.squeeze(-1)
        
        # Dot product with local z-axis [0, 0, 1]
        alignment = up_vector_local[:, 2]  # z-component
        orientation_error = 1.0 - alignment
        rewards["orientation"] = orientation_error * self.reward_scales["orientation"]
        
        # 6. Feet air time (encourages walking gait)
        device = base_lin_vel.device
        self._feet_air_time = self._feet_air_time.to(device)
        
        # Update air time counter based on contact
        self._feet_air_time = torch.where(
            feet_contact > 0.5,
            torch.zeros_like(self._feet_air_time),
            self._feet_air_time + self._scene.physics_sim_view.get_physics_dt()
        )
        
        # Calculate air time reward (sigmoid function to saturate reward)
        air_time_reward = torch.sum(
            torch.clip(self._feet_air_time, 0.0, 0.5) / 0.5, dim=1
        )
        rewards["feet_air_time"] = air_time_reward * self.reward_scales["feet_air_time"]
        
        # 7. Foot contact reward (small reward for making contact)
        contact_reward = torch.sum(feet_contact, dim=1)
        rewards["feet_contact"] = contact_reward * self.reward_scales["feet_contact"]
        
        # Total reward is sum of all reward terms
        rewards["total"] = sum(rewards.values())
        
        return rewards


class LocomotionResetManager(ResetManager):
    """Reset manager for locomotion tasks."""
    
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
        self.vel_noise = 0.1  # Velocity noise
        
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
        
        # Randomize command velocity
        obs_manager = self._scene.env_mgmt._observation_manager
        
        # Randomize forward velocity command in range [0.5, 1.5] m/s
        cmd_vel_x = np.random.uniform(0.5, 1.5)
        cmd_vel_y = np.random.uniform(-0.3, 0.3)
        cmd_vel_yaw = np.random.uniform(-0.3, 0.3)
        
        obs_manager.command_velocity = torch.tensor(
            [[cmd_vel_x, cmd_vel_y, cmd_vel_yaw]],
            dtype=torch.float32
        )
        
        # Reset feet air time in reward manager
        reward_manager = self._scene.env_mgmt._reward_manager
        reward_manager._feet_air_time = torch.zeros((1, 4), dtype=torch.float32)
        reward_manager._prev_joint_vel = None
        reward_manager._prev_actions = None


class LocomotionTerminationManager(TerminationManager):
    """Termination manager for locomotion tasks."""
    
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
        self.max_episode_length = 5.0  # Maximum episode length in seconds
        self.max_height = 1.0          # Maximum height of the base above ground
        self.min_height = 0.2          # Minimum height of the base above ground
        self.max_roll = 1.0            # Maximum roll angle in radians
        self.max_pitch = 1.0           # Maximum pitch angle in radians
        
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
        
        # Convert quaternion to Euler angles
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        roll, pitch, yaw = quat_to_euler_angles(quat)
        
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
        terminations["time_limit"] = time_limit
        terminations["base_termination"] = base_termination
        
        # Combined termination signal
        terminations["terminated"] = torch.logical_or(time_limit, base_termination)
        
        return terminations


if __name__ == "__main__":
    main() 