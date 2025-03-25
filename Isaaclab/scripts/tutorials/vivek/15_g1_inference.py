#!/usr/bin/env python3

"""
Tutorial 15: G1 Robot Navigation Inference
=========================================

This tutorial demonstrates how to use a trained navigation policy 
(or a simulated one) to make a G1 robot navigate to specific objects
in a predefined sequence. Key features include:

1. Setting up a scene with multiple target objects
2. Sequential navigation to each target
3. Visualization of robot path and targets
4. Simple object interaction logic

This provides a practical example of applying navigation capabilities
in a structured task scenario.
"""

import math
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import random

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
from pxr import UsdGeom, Gf, Sdf


def main():
    """Main function for the G1 navigation inference tutorial."""
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
    
    # Define target positions - these will have objects placed at them
    target_positions = [
        (3.0, 3.0, 0.0),    # Position 1: front-right
        (-3.0, 3.0, 0.0),   # Position 2: front-left
        (-3.0, -3.0, 0.0),  # Position 3: back-left
        (3.0, -3.0, 0.0),   # Position 4: back-right
        (0.0, 0.0, 0.0),    # Position 5: center (return home)
    ]
    
    # Create target objects (colored cubes) and store their paths
    target_objects = []
    for i, pos in enumerate(target_positions[:-1]):  # Skip the last position (home)
        obj_path = create_target_object(scene, f"target_{i}", pos, i)
        target_objects.append(obj_path)
    
    # Create the G1 navigation environment
    env = G1NavigationEnv(scene)
    
    # Run the simulation
    sim_context.reset()
    env.reset()
    
    # Print instructions
    print("=" * 80)
    print("G1 Robot Navigation Inference Demo")
    print("=" * 80)
    print("This demo shows a G1 robot navigating to specific objects in sequence.")
    print("The robot will visit each colored cube and then return to the center.")
    
    # Initialize path visualization
    path_tracer = init_path_visualization(scene)
    prev_pos = None
    
    # Set up simulation state
    current_target_idx = 0
    target_reached_counter = 0
    
    # Set initial target
    update_current_target(env, target_positions[current_target_idx])
    print(f"Robot is heading to target {current_target_idx + 1} of {len(target_positions)}")
    
    # Run simulation with navigation to each target in sequence
    timer = Timer()
    timer.start()
    simulation_duration = 180.0  # 3 minutes total simulation time
    
    # For visualization of current target
    current_target_indicator = create_target_indicator(scene)
    
    # Main simulation loop
    while timer.time() < simulation_duration:
        # Get current observations
        obs_dict = env._observation_manager.compute_observations()
        goal_pos_local = obs_dict["goal_pos_local"][0]  # Local coordinates of goal
        robot_pos = obs_dict["base_pos"][0]
        dist_to_goal = obs_dict["dist_to_goal"][0].item()
        
        # Update target visualization
        goal_pos_world = obs_dict["goal_pos_world"][0]
        update_target_indicator(current_target_indicator, goal_pos_world)
        
        # Update path visualization
        update_path_visualization(path_tracer, robot_pos, prev_pos)
        prev_pos = robot_pos.clone()
        
        # Compute navigation actions
        actions = compute_navigation_actions(goal_pos_local)
        actions = to_torch(actions, device="cpu", dtype=torch.float32)
        
        # Step the environment
        obs, rewards, dones, infos = env.step(actions)
        
        # Check if we've reached the current target
        if dist_to_goal < 0.5:  # Within half a meter of target
            target_reached_counter += 1
            
            # We need to maintain proximity for a bit to consider it "reached"
            if target_reached_counter >= 20:  # About 0.2 seconds at 10ms per step
                # "Collect" the object by changing its color when reached
                if current_target_idx < len(target_positions) - 1:  # Skip home position
                    object_collected_effect(target_objects[current_target_idx])
                
                # Move to the next target
                current_target_idx = (current_target_idx + 1) % len(target_positions)
                update_current_target(env, target_positions[current_target_idx])
                
                # Print progress message
                if current_target_idx == 0:
                    print("Route completed! Starting again...")
                else:
                    print(f"Target reached! Moving to target {current_target_idx + 1} of {len(target_positions)}")
                
                # Reset counter
                target_reached_counter = 0
        else:
            # Reset counter if we're not at the target
            target_reached_counter = 0
        
        # Step the simulation
        sim_context.step()
        
        # Sleep to maintain real-time
        time.sleep(0.01)
    
    # Clean up
    env.close()
    simulation_app.close()


def create_target_object(scene: InteractiveScene, name: str, position: Tuple[float, float, float], 
                         index: int) -> str:
    """Create a target object (colored cube) at the specified position.
    
    Args:
        scene: The interactive scene.
        name: Name of the target object.
        position: (x, y, z) position of the target.
        index: Index of the target (used for color variation).
        
    Returns:
        The path to the created object.
    """
    # Create a cube to represent the target
    cube_path = f"/World/{name}"
    
    # Create the cube
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Cube",
        prim_path=cube_path,
        attributes={"size": 0.5}  # 0.5m cube
    )
    
    # Position the cube
    x, y, z = position
    omni.kit.commands.execute(
        "TransformPrimCommand",
        path=cube_path,
        translation=Gf.Vec3d(x, y, z + 0.25)  # Offset in z to place on ground
    )
    
    # Add a unique color based on index
    colors = [
        (1.0, 0.2, 0.2),  # Red
        (0.2, 1.0, 0.2),  # Green
        (0.2, 0.2, 1.0),  # Blue
        (1.0, 1.0, 0.2),  # Yellow
    ]
    color_idx = index % len(colors)
    
    # Create material
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Material",
        prim_path=f"{cube_path}/Material",
        attributes={}
    )
    
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Shader",
        prim_path=f"{cube_path}/Material/Shader",
        attributes={"info:id": "UsdPreviewSurface"}
    )
    
    omni.kit.commands.execute(
        "SetAttributeCommand",
        attr_path=f"{cube_path}/Material/Shader.inputs:diffuseColor",
        value=Gf.Vec3f(*colors[color_idx])
    )
    
    omni.kit.commands.execute(
        "ConnectAttribute",
        source_path=f"{cube_path}/Material/Shader.outputs:surface",
        target_path=f"{cube_path}/Material.outputs:surface"
    )
    
    omni.kit.commands.execute(
        "BindMaterial",
        material_path=f"{cube_path}/Material",
        prim_path=cube_path
    )
    
    return cube_path


def object_collected_effect(object_path: str) -> None:
    """Apply a visual effect to show an object has been collected.
    
    Args:
        object_path: Path to the object to modify.
    """
    # Make the object semi-transparent to indicate it's been collected
    try:
        omni.kit.commands.execute(
            "SetAttributeCommand",
            attr_path=f"{object_path}/Material/Shader.inputs:opacity",
            value=0.3
        )
        
        # Add a slight animation by raising it a bit
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(object_path)
        
        # Get current position
        xform = UsdGeom.Xformable(prim)
        current_transform = xform.GetLocalTransformation()
        translation = current_transform.ExtractTranslation()
        
        # Raise it a bit
        new_translation = Gf.Vec3d(translation[0], translation[1], translation[2] + 0.5)
        
        omni.kit.commands.execute(
            "TransformPrimCommand",
            path=object_path,
            translation=new_translation
        )
    except Exception as e:
        print(f"Error applying collection effect: {e}")


def create_target_indicator(scene: InteractiveScene) -> str:
    """Create a visual indicator for the current target location.
    
    Args:
        scene: The interactive scene to add the indicator to.
        
    Returns:
        The path to the created indicator.
    """
    # Create a sphere to indicate the target
    sphere_path = "/World/current_target_indicator"
    omni.kit.commands.execute(
        "CreatePrimCommand",
        prim_type="Sphere",
        prim_path=sphere_path,
        attributes={"radius": 0.2}
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
        value=Gf.Vec3f(1.0, 1.0, 0.0)  # Yellow
    )
    
    # Make it slightly transparent
    omni.kit.commands.execute(
        "SetAttributeCommand",
        attr_path=f"{sphere_path}/Material/Shader.inputs:opacity",
        value=0.8
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
    pos = Gf.Vec3d(position[0].item(), position[1].item(), position[2].item() + 0.5)  # Offset above the ground
    
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
    
    This simulates a trained policy by implementing a simple differential 
    drive controller that converts goal position to wheel velocities.
    
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


def update_current_target(env: Any, position: Tuple[float, float, float]) -> None:
    """Update the robot's current navigation target.
    
    Args:
        env: The navigation environment.
        position: (x, y, z) position of the new target.
    """
    target_pos = torch.tensor([[position[0], position[1], position[2]]])
    env._observation_manager.goal_position = target_pos


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
        self.goal_position = torch.tensor([[0.0, 0.0, 0.0]])  # Default goal is center
        
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
        
        # Define reward terms and their weights (not used in inference mode)
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
        # For inference, we don't actually need to compute rewards
        # but we'll return a minimal set for compatibility
        device = self.robot.device
        return {
            "total": torch.zeros(1, dtype=torch.float32, device=device)
        }


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
        
    def reset(self) -> None:
        """Reset the environment to initial conditions."""
        # Reset wheel velocities to zero
        joint_velocities = {
            "left_wheel": 0.0,
            "right_wheel": 0.0
        }
        self.robot.set_joint_velocities(joint_velocities)
        
        # Set base pose to center of the scene
        self.robot.set_world_pose(
            position=(0.0, 0.0, 0.1),
            orientation=(0, 0, 0)
        )
        
        # Reset base velocity to zero
        self.robot.set_body_velocity(
            linear_velocity=(0, 0, 0),
            angular_velocity=(0, 0, 0)
        )
        
        # Reset progress tracking in reward manager
        reward_manager = self._scene.env_mgmt._reward_manager
        reward_manager._prev_goal_distance = None


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
        
    def get_terminations(self) -> Dict[str, torch.Tensor]:
        """Get termination signals for the current state.
        
        Returns:
            A dictionary of termination signals.
        """
        # In inference mode, we don't want to terminate
        device = self.robot.device
        return {
            "terminated": torch.zeros(1, dtype=torch.bool, device=device)
        }


if __name__ == "__main__":
    main() 