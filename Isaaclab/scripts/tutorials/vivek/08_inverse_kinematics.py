# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use inverse kinematics to control a robot arm.

This tutorial helps beginners understand:
1. How to set up differential inverse kinematics for a robot
2. How to control a robot's end-effector in Cartesian space
3. How to make a robot follow a trajectory in 3D space

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/08_inverse_kinematics.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create an argument parser
parser = argparse.ArgumentParser(description="Tutorial on using inverse kinematics")
# Add standard AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Once the app is launched, we can import the rest of the modules and define our simulation."""

import numpy as np
import math
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, ArticulationCfg
from isaaclab.scene.ground import GroundPlaneCfg
from isaaclab.scene.visual import SphereCfg
from isaaclab.controllers.ik.differential_ik import DifferentialIK, DifferentialIKCfg
from isaaclab.utils.math import quat_from_euler_xyz


def main():
    """Main function that demonstrates inverse kinematics for robot control."""
    
    print("Welcome to Isaac Lab! Let's learn about inverse kinematics.")
    
    # Step 1: Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # Step 2: Set up a camera view appropriate for viewing the robot
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.5])
    
    # Step 3: Create a scene
    scene = InteractiveScene(sim)
    
    # Step 4: Add a ground plane
    ground_cfg = GroundPlaneCfg(
        size=np.array([10.0, 10.0]),
        color=np.array([0.2, 0.2, 0.2, 1.0])
    )
    scene.add_ground(ground_cfg)
    
    # Step 5: Add a Franka Panda robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/robot",
        name="panda_robot",
        usd_path="{ISAACLAB_ASSETS}/robots/franka_panda/panda.usd",
        init_state={
            "pos": np.array([0.0, 0.0, 0.0]),
            "dof_pos": np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0, 0.04, 0.04])
        }
    )
    robot = scene.add_articulation(robot_cfg)
    
    # Step 6: Set up the differential IK controller
    # Get the robot's degrees of freedom (excluding the gripper)
    robot_dof = robot.num_dof - 2  # Subtract 2 for the gripper joints
    
    # Configure the differential IK controller
    ik_cfg = DifferentialIKCfg(
        position_gain=1.0,            # Gain for position control
        rotation_gain=1.0,            # Gain for orientation control
        damping=0.05,                 # Damping factor for stability
        step_size=1.0,                # Step size for the IK solver
        use_nullspace_projection=True # Use nullspace for secondary objectives
    )
    
    # Create the IK controller
    ik_controller = DifferentialIK(
        cfg=ik_cfg,
        robot_articulation=robot,
        end_effector_prim_path=f"{robot.prim_path}/panda_hand",
        ik_control_dof_indices=list(range(robot_dof))  # Control the arm joints, not the gripper
    )
    
    # Step 7: Add visual markers for the target positions
    # We'll create a series of positions that form a circle in 3D space
    num_waypoints = 20
    radius = 0.3
    center_position = np.array([0.3, 0.0, 0.5])  # Center of the circle
    target_markers = []
    waypoints = []
    
    for i in range(num_waypoints):
        # Calculate position on a circle
        angle = 2.0 * math.pi * i / num_waypoints
        x = center_position[0] + radius * math.cos(angle)
        y = center_position[1] + radius * math.sin(angle)
        z = center_position[2]
        
        # Store waypoint
        position = np.array([x, y, z])
        waypoints.append(position)
        
        # Create a small sphere to visualize the waypoint
        marker_cfg = SphereCfg(
            prim_path=f"/World/target_marker_{i}",
            radius=0.02,
            color=np.array([1.0, 0.5, 0.0, 1.0]),  # Orange
            position=position
        )
        marker = scene.add_visual_sphere(marker_cfg)
        target_markers.append(marker)
    
    # Step 8: Reset the scene to initialize physics
    scene.reset()
    print("[INFO]: Setup complete!")
    
    # Get the initial end-effector pose
    current_ee_pos, current_ee_rot = robot.get_ee_transform()
    print(f"Initial end-effector position: {current_ee_pos}")
    
    # Step 9: Run the simulation and control the robot with IK
    step_count = 0
    waypoint_idx = 0
    dwell_steps = 20  # Stay at each waypoint for 0.2 seconds
    dwell_counter = 0
    
    # Target pose variables
    target_position = waypoints[0]
    # We'll maintain a constant orientation pointing downward
    target_orientation = quat_from_euler_xyz(np.array([math.pi, 0.0, 0.0]))
    
    while simulation_app.is_running() and step_count < 2000:  # Run for ~20 seconds
        # Get current end-effector pose
        current_ee_pos, current_ee_rot = robot.get_ee_transform()
        
        # Display info every 100 steps
        if step_count % 100 == 0:
            print(f"\nStep {step_count}:")
            print(f"  Current EE position: {current_ee_pos}")
            print(f"  Target position: {target_position}")
            
            # Calculate distance to target
            distance = np.linalg.norm(current_ee_pos - target_position)
            print(f"  Distance to target: {distance:.4f} meters")
        
        # Check if we should move to the next waypoint
        if dwell_counter >= dwell_steps:
            # Move to the next waypoint
            waypoint_idx = (waypoint_idx + 1) % num_waypoints
            target_position = waypoints[waypoint_idx]
            
            print(f"\nMoving to waypoint {waypoint_idx + 1}/{num_waypoints}")
            print(f"Target position: {target_position}")
            
            dwell_counter = 0
        
        # Apply IK to move towards the target
        joint_positions = ik_controller.compute_inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation
        )
        
        # Set joint positions for the arm (not the gripper)
        robot_joint_positions = robot.get_joint_positions()
        robot_joint_positions[:robot_dof] = joint_positions
        robot.set_joint_positions(robot_joint_positions)
        
        # Increment counters
        dwell_counter += 1
        step_count += 1
        
        # Perform one simulation step
        scene.step()
    
    print("Simulation complete! You've used inverse kinematics to control a robot arm.")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation application when done
    simulation_app.close() 