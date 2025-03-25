#!/usr/bin/env python3

"""
Tutorial 14: G1 Robot Navigation
===============================

This tutorial demonstrates how to create a manager-based environment 
for the G1 robot to navigate to specific coordinates. We adapt the
navigation logic from the previous tutorial but for a different robot:

1. Using the G1 mobile robot instead of a quadruped
2. Implementing differential drive control for wheeled navigation
3. Adapting the coordinate navigation system for the G1's capabilities
4. Visualizing navigation targets and robot trajectory

This provides a practical example of how the same navigation
concepts can be adapted for different robot platforms.
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
    """Main function for the G1 navigation tutorial."""
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
    env = G1NavigationEnv(scene)
    
    # Run the simulation
    sim_context.reset()
    env.reset()
    
    # Print instructions
    print("=" * 80)
    print("G1 Robot Navigation Tutorial")
    print("=" * 80)
    print("This demo shows a G1 robot navigating to target locations.")
    print("The robot will try to reach randomly placed targets in the environment.")
    print("Green sphere indicates the current target location.")
    
    # Create visualization for target
    target_sphere = create_target_indicator(scene)
    
    # Track robot path for visualization
    path_tracer = init_path_visualization(scene)
    prev_pos = None
    
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
        
        # Update path visualization
        robot_pos = obs_dict["base_pos"][0]
        update_path_visualization(path_tracer, robot_pos, prev_pos)
        prev_pos = robot_pos.clone()
        
        # Simple navigation controller: convert local goal position to velocity commands
        actions = compute_navigation_actions(goal_pos_local)
        actions = to_torch(actions, device="cpu", dtype=torch.float32)
        
        # Step the environment
        obs, rewards, dones, infos = env.step(actions)
        
        # Reset if episode is done
        if dones.any():
            env.reset()
            # Clear path visualization when resetting
            path_tracer = init_path_visualization(scene, reset_existing=path_tracer)
            prev_pos = None
            
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


def init_path_visualization(scene: InteractiveScene, reset_existing: Optional[str] = None) -> str:
    """Initialize or reset the path visualization for the robot.
    
    Args:
        scene: The interactive scene.
        reset_existing: Path to existing visualization to reset, if any.
        
    Returns:
        The path to the created path tracer.
    """
    # If resetting existing visualization, delete it first
    if reset_existing:
        try:
            omni.kit.commands.execute("DeletePrimCommand", prim_path=reset_existing)
        except:
            pass
    
    # Create a new visualization
    path_tracer = "/World/path_tracer"
    
    # Create basis curves prim for path
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="BasisCurves",
        prim_path=path_tracer,
        attributes={}
    )
    
    # Initialize with empty points
    omni.kit.commands.execute(
        "SetAttributeCommand",
        attr_path=f"{path_tracer}.points",
        value=[]
    )
    
    # Set curve type to linear
    omni.kit.commands.execute(
        "SetAttributeCommand",
        attr_path=f"{path_tracer}.type",
        value="linear"
    )
    
    # Set curve width
    omni.kit.commands.execute(
        "SetAttributeCommand",
        attr_path=f"{path_tracer}.widths",
        value=[0.05]  # 5cm wide line
    )
    
    # Set curve color to blue
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Material",
        prim_path=f"{path_tracer}/Material",
        attributes={}
    )
    
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Shader",
        prim_path=f"{path_tracer}/Material/Shader",
        attributes={"info:id": "UsdPreviewSurface"}
    )
    
    omni.kit.commands.execute(
        "SetAttributeCommand",
        attr_path=f"{path_tracer}/Material/Shader.inputs:diffuseColor",
        value=Gf.Vec3f(0.0, 0.2, 1.0)  # Blue color
    )
    
    omni.kit.commands.execute(
        "ConnectAttribute",
        source_path=f"{path_tracer}/Material/Shader.outputs:surface",
        target_path=f"{path_tracer}/Material.outputs:surface"
    )
    
    omni.kit.commands.execute(
        "BindMaterial",
        material_path=f"{path_tracer}/Material",
        prim_path=path_tracer
    )
    
    return path_tracer


def update_path_visualization(path_tracer: str, current_pos: torch.Tensor, prev_pos: torch.Tensor = None) -> None:
    """Update the path visualization with the robot's current position.
    
    Args:
        path_tracer: The path to the path tracer prim.
        current_pos: The current position of the robot.
        prev_pos: The previous position of the robot.
    """
    if prev_pos is None:
        return  # Nothing to draw yet
    
    # Get the current points
    try:
        from pxr import Usd, Vt
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path_tracer)
        points_attr = prim.GetAttribute("points")
        existing_points = points_attr.Get() or Vt.Vec3fArray([])
        
        # Convert tensor positions to Vt.Vec3f
        p1 = Gf.Vec3f(prev_pos[0].item(), prev_pos[1].item(), prev_pos[2].item() + 0.1)  # Slightly above ground
        p2 = Gf.Vec3f(current_pos[0].item(), current_pos[1].item(), current_pos[2].item() + 0.1)
        
        # Add new points if they're far enough apart
        min_distance = 0.05  # Minimum distance between points
        if len(existing_points) == 0 or (p1 - Gf.Vec3f(existing_points[-1])).GetLength() > min_distance:
            # Add new line segment (prev_pos to current_pos)
            new_points = Vt.Vec3fArray(list(existing_points) + [p1, p2])
            
            # Update the points
            points_attr.Set(new_points)
            
            # Update curve topology (vertex counts)
            vertex_counts = [2] * (len(new_points) // 2)
            prim.GetAttribute("curveVertexCounts").Set(vertex_counts)
    except Exception as e:
        print(f"Error updating path visualization: {e}")


def compute_navigation_actions(goal_pos_local: torch.Tensor) -> np.ndarray:
    """Compute actions to navigate the G1 robot to the goal position.
    
    This implements a simple differential drive controller that converts
    goal position to wheel velocities.
    
    Args:
        goal_pos_local: The goal position in the robot's local frame.
        
    Returns:
        An array of actions for the robot's wheel velocities.
    """
    # Extract local goal coordinates
    goal_x = goal_pos_local[0].item()  # Forward direction
    goal_y = goal_pos_local[1].item()  # Left direction
    goal_dist = np.sqrt(goal_x**2 + goal_y**2)
    
    # Simple differential drive control
    max_speed = 1.0  # Maximum wheel speed
    
    # Compute forward velocity and turning rate
    forward_speed = goal_x
    turning_rate = np.arctan2(goal_y, max(0.1, abs(goal_x)))  # Angle to goal
    
    # Convert to left and right wheel velocities using differential drive equations
    left_wheel = forward_speed - turning_rate
    right_wheel = forward_speed + turning_rate
    
    # Scale wheel velocities to respect max_speed
    max_wheel = max(abs(left_wheel), abs(right_wheel))
    if max_wheel > max_speed:
        scale = max_speed / max_wheel
        left_wheel *= scale
        right_wheel *= scale
    
    # When close to goal, slow down
    slowdown_distance = 0.5
    if goal_dist < slowdown_distance:
        slow_factor = goal_dist / slowdown_distance
        left_wheel *= slow_factor
        right_wheel *= slow_factor
    
    # Return wheel velocities as action
    return np.array([[left_wheel, right_wheel]])


class G1NavigationEnv(VecEnvMT):
    """A manager-based environment for G1 robot navigation tasks."""
    
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
        
        self._observation_manager = G1ObservationManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._reward_manager = G1RewardManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._reset_manager = G1ResetManager(
            scene=self._scene,
            scene_manager=self._scene_manager,
        )
        
        self._termination_manager = G1TerminationManager(
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
        """Create the scene entity configuration for the G1 navigation environment.
        
        Returns:
            A dictionary of scene entity configurations.
        """
        # Configure the G1 robot
        cfg = {
            "robot": SceneEntityCfg(
                # Use the G1 robot from Isaac Lab
                "omni.isaac.lab.assets.carter.G1",
                # Robot spawn pose (higher Z as G1 is taller than a quadruped)
                spawn={"pos": (0, 0, 0.1), "rot": (0, 0, 0)},
                # No initial joint positions needed for G1 as it uses differential drive
                init_state={},
                # Keep default physics properties
                rigid_props={"disable_gravity": False},
            ),
        }
        return cfg
    
    @property
    def action_dim(self) -> int:
        """Return the dimension of the action space."""
        # 2 for [left_wheel_velocity, right_wheel_velocity]
        return 2


class G1ObservationManager(ObservationManager):
    """Observation manager for G1 navigation tasks."""
    
    def __init__(self, scene: InteractiveScene, scene_manager: SceneEntityManager):
        """Initialize the observation manager.
        
        Args:
            scene: The interactive scene.
            scene_manager: The scene entity manager.
        """
        super().__init__(scene, scene_manager)
        
        # Get robot reference
        self.robot = scene_manager.entity_map["robot"]
        
        # Navigation goal parameters
        self.goal_position = torch.tensor([[5.0, 0.0, 0.0]])  # Default goal
        
        # Define observation groups
        self._observation_groups = {
            "policy": [
                "base_pos",         # Base position
                "base_quat",        # Base orientation
                "base_lin_vel",     # Base linear velocity
                "base_ang_vel",     # Base angular velocity
                "wheel_velocities", # Current wheel velocities
                "goal_pos_local",   # Goal position in robot local frame
                "heading_to_goal",  # Heading angle to goal
                "dist_to_goal",     # Distance to goal
            ]
        }
        
        # Store wheel joint names for G1 robot
        self.wheel_joint_names = ["left_wheel", "right_wheel"]
        
    def compute_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for the current state.
        
        Returns:
            A dictionary of observation tensors.
        """
        observations = {}
        
        # Get base states
        base_pos, base_quat = self.robot.get_world_pose()
        base_lin_vel, base_ang_vel = self.robot.get_body_velocity()
        
        # Get wheel velocities (important for a differential drive robot)
        wheel_velocities = torch.zeros((1, 2), device=base_pos.device)
        for i, joint_name in enumerate(self.wheel_joint_names):
            wheel_velocities[0, i] = self.robot.get_joint_velocities({joint_name: 0.0})[joint_name]
        
        # Ensure goal position is on the same device as base_pos
        goal_pos_world = self.goal_position.to(base_pos.device)
        
        # Compute goal position in robot's local frame
        goal_pos_relative = goal_pos_world - base_pos
        goal_pos_local = quat_rotate_inverse(base_quat, goal_pos_relative)
        
        # Compute distance to goal (XY plane distance only)
        dist_to_goal = torch.norm(goal_pos_relative[:, :2], dim=1, keepdim=True)
        
        # Compute heading to goal (angle in the x-y plane)
        heading_to_goal = torch.atan2(goal_pos_local[:, 1], goal_pos_local[:, 0]).unsqueeze(1)
        
        # Store all observations
        observations["base_pos"] = base_pos
        observations["base_quat"] = base_quat
        observations["base_lin_vel"] = base_lin_vel
        observations["base_ang_vel"] = base_ang_vel
        observations["wheel_velocities"] = wheel_velocities
        observations["goal_pos_world"] = goal_pos_world  # For visualization
        observations["goal_pos_local"] = goal_pos_local
        observations["heading_to_goal"] = heading_to_goal
        observations["dist_to_goal"] = dist_to_goal
        
        return observations
    
    def process_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process wheel velocity commands for the G1 robot.
        
        Args:
            actions: The wheel velocity commands [left_wheel, right_wheel].
            
        Returns:
            A dictionary of processed actions.
        """
        # Extract wheel velocities
        left_wheel_vel = actions[:, 0]
        right_wheel_vel = actions[:, 1]
        
        # Scale to appropriate velocity range for G1
        # G1 uses direct velocity control, values are in rad/s
        max_wheel_vel = 10.0  # Maximum wheel velocity in rad/s
        
        left_wheel_vel = left_wheel_vel * max_wheel_vel
        right_wheel_vel = right_wheel_vel * max_wheel_vel
        
        # Set wheel velocity targets
        joint_velocities = {
            "left_wheel": left_wheel_vel.item(),
            "right_wheel": right_wheel_vel.item()
        }
        
        self.robot.set_joint_velocities(joint_velocities)
        
        return {"wheel_velocities": actions}


class G1RewardManager(RewardManager):
    """Reward manager for G1 navigation tasks."""
    
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
            "forward_velocity": 0.5,     # Reward for moving toward the goal
            "goal_progress": 1.0,        # Reward for approaching the goal
            "goal_reached": 10.0,        # Bonus for reaching the goal
            "heading_alignment": 0.2,    # Reward for facing toward the goal
            "energy_efficiency": -0.001, # Penalty for energy usage (wheel speeds)
            "excessive_rotation": -0.1   # Penalty for excessive rotation
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
        base_lin_vel, base_ang_vel = self.robot.get_body_velocity()
        
        # Get observation manager for navigation-specific data
        obs_manager = self._scene.env_mgmt._observation_manager
        observations = obs_manager.compute_observations()
        goal_pos_local = observations["goal_pos_local"]
        dist_to_goal = observations["dist_to_goal"]
        heading_to_goal = observations["heading_to_goal"]
        wheel_velocities = observations["wheel_velocities"]
        
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
        
        # 3. Heading alignment reward (facing toward goal)
        # Reward is maximized when heading directly toward the goal
        heading_alignment = torch.cos(heading_to_goal)
        rewards["heading_alignment"] = heading_alignment * self.reward_scales["heading_alignment"]
        
        # 4. Forward velocity reward
        # Reward for moving in the direction of the goal
        forward_vel_reward = base_lin_vel[:, 0] * torch.cos(heading_to_goal.squeeze())
        rewards["forward_velocity"] = forward_vel_reward * self.reward_scales["forward_velocity"]
        
        # 5. Energy efficiency penalty
        # Penalize high wheel speeds to encourage efficient motion
        energy_usage = torch.sum(wheel_velocities**2, dim=1)
        rewards["energy_efficiency"] = energy_usage * self.reward_scales["energy_efficiency"]
        
        # 6. Excessive rotation penalty
        # Penalize spinning in place too much
        excessive_rotation = torch.abs(base_ang_vel[:, 2])  # Yaw rate
        rewards["excessive_rotation"] = excessive_rotation * self.reward_scales["excessive_rotation"]
        
        # Total reward is sum of all reward terms
        rewards["total"] = sum(rewards.values())
        
        return rewards


class G1ResetManager(ResetManager):
    """Reset manager for G1 navigation tasks."""
    
    def __init__(self, scene: InteractiveScene, scene_manager: SceneEntityManager):
        """Initialize the reset manager.
        
        Args:
            scene: The interactive scene.
            scene_manager: The scene entity manager.
        """
        super().__init__(scene, scene_manager)
        
        # Get robot from scene manager
        self.robot = scene_manager.entity_map["robot"]
        
        # Reset noise parameters
        self.pos_noise = 0.2  # Position noise in meters
        self.rot_noise = 0.2  # Rotation noise in radians
        
        # Navigation parameters
        self.min_goal_dist = 3.0   # Minimum distance to goal
        self.max_goal_dist = 8.0   # Maximum distance to goal
        self.goal_z = 0.0          # Height of the goal (ground level)
        
    def reset(self) -> None:
        """Reset the environment to initial conditions with randomization."""
        # Reset wheel velocities to zero
        joint_velocities = {
            "left_wheel": 0.0,
            "right_wheel": 0.0
        }
        self.robot.set_joint_velocities(joint_velocities)
        
        # Reset base position with small random offset
        x_noise = np.random.uniform(-self.pos_noise, self.pos_noise)
        y_noise = np.random.uniform(-self.pos_noise, self.pos_noise)
        
        # Set base pose (G1 height is about 0.1m above ground)
        self.robot.set_world_pose(
            position=(x_noise, y_noise, 0.1),
            orientation=(0, 0, np.random.uniform(-self.rot_noise, self.rot_noise))
        )
        
        # Reset base velocity to zero
        self.robot.set_body_velocity(
            linear_velocity=(0, 0, 0),
            angular_velocity=(0, 0, 0)
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
        robot_pos, _ = self.robot.get_world_pose()
        
        # Generate random distance and angle
        distance = np.random.uniform(self.min_goal_dist, self.max_goal_dist)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Calculate goal position in world frame
        goal_x = robot_pos[0, 0].item() + distance * np.cos(angle)
        goal_y = robot_pos[0, 1].item() + distance * np.sin(angle)
        
        # Set the goal position (at ground level)
        goal_position = torch.tensor([[goal_x, goal_y, self.goal_z]])
        obs_manager.goal_position = goal_position


class G1TerminationManager(TerminationManager):
    """Termination manager for G1 navigation tasks."""
    
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
        self.max_episode_length = 30.0  # Maximum episode length in seconds
        self.goal_threshold = 0.5       # Distance threshold to consider goal reached
        
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
        
        # Get goal information
        obs_manager = self._scene.env_mgmt._observation_manager
        observations = obs_manager.compute_observations()
        dist_to_goal = observations["dist_to_goal"]
        
        # Goal reached termination
        goal_reached = (dist_to_goal < self.goal_threshold)
        
        # Time limit termination
        time_limit = torch.tensor([self._episode_time > self.max_episode_length], 
                                  dtype=torch.bool, device=device)
        
        # Store termination signals
        terminations["goal_reached"] = goal_reached
        terminations["time_limit"] = time_limit
        
        # Combined termination signal
        terminations["terminated"] = torch.logical_or(goal_reached, time_limit)
        
        return terminations


if __name__ == "__main__":
    main() 