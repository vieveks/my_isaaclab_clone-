# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to work with a robotic arm in Isaac Lab.

This tutorial helps beginners understand:
1. How to spawn a robotic arm (Franka Emika Panda)
2. How to control the arm's joint positions
3. How to read joint states and end-effector pose

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/04_robotic_arm.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on working with a robotic arm")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip


def design_scene():
    """Designs the scene by spawning ground plane, light, and a Franka Panda robot."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # spawn dome light for better visibility
    cfg_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.8, 0.8, 0.8),
    )
    cfg_light.func("/World/Light", cfg_light)
    
    # Create a parent for the robot
    prim_utils.create_prim("/World/Robots", "Xform")
    
    # Create a Franka Panda robot
    robot_cfg = FRANKA_PANDA_CFG.copy()
    robot_cfg.prim_path = "/World/Robots/franka_panda"
    robot = Articulation(cfg=robot_cfg)
    
    # Return the scene entities
    scene_entities = {"robot": robot}
    return scene_entities


def main():
    """Main function."""
    
    print("Welcome to Isaac Lab! Let's work with a robot arm.")
    
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set up a camera view appropriate for viewing the robot
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    
    # Design scene by adding assets to it
    scene_entities = design_scene()
    
    # Extract the robot for convenience
    robot = scene_entities["robot"]
    
    # Play the simulator
    sim.reset()
    
    # Reset robot state
    robot.reset()
    
    print("[INFO]: Simulation setup complete with robot arm!")
    
    # Get information about the robot
    num_dof = len(robot.data.joint_pos[0])  # Number of DOFs is the length of joint positions
    dof_names = robot.data.joint_names[0]  # Access joint names from robot.data.joint_names
    print(f"Robot has {num_dof} degrees of freedom")
    print(f"Joint names: {dof_names}")
    
    # Find the body index for the end effector (panda_hand)
    ee_body_name = "panda_hand"
    ee_body_index = robot.find_bodies(ee_body_name)[0][0]
    
    # Set up a list of target positions for the robot joints
    # These values are specific to the Franka Panda robot
    target_positions = [
        # Format: [joint1, joint2, ..., joint7, finger1, finger2]
        torch.tensor([0.0, -0.4, 0.0, -2.0, 0.0, 2.0, 0.8, 0.0, 0.0], device=sim.device),  # Home with closed gripper
        torch.tensor([0.0, -0.4, 0.0, -2.0, 0.0, 2.0, 0.8, 0.04, 0.04], device=sim.device),  # Home with open gripper
        torch.tensor([0.5, -0.2, 0.0, -1.5, 0.0, 1.5, 0.0, 0.04, 0.04], device=sim.device),  # Different arm pose
        torch.tensor([-0.5, -0.2, 0.0, -1.5, 0.0, 1.5, 0.0, 0.0, 0.0], device=sim.device),   # Another pose with closed gripper
        torch.tensor([0.0, -0.4, 0.0, -2.0, 0.0, 2.0, 0.8, 0.0, 0.0], device=sim.device),    # Back to home
    ]
    
    # Define simulation parameters
    sim_dt = sim.get_physics_dt()
    current_target_idx = 0
    dwell_steps = 200  # Stay at each position for 2 seconds
    dwell_counter = 0
    step_count = 0
    
    # Simulate physics
    while simulation_app.is_running() and step_count < 2000:  # Run for ~20 seconds
        # Get the current robot state
        if step_count % 100 == 0:
            current_position = robot.data.joint_pos.clone().cpu().numpy()[0]
            # Get end-effector position and rotation from body_state_w
            ee_state = robot.data.body_state_w[0, ee_body_index]
            ee_position = ee_state[:3].cpu().numpy()
            ee_rotation = ee_state[3:7].cpu().numpy()
            print(f"Step {step_count}:")
            print(f"  End-effector position: {ee_position}")
            
        # Determine if we need to change targets
        if dwell_counter >= dwell_steps:
            current_target_idx = (current_target_idx + 1) % len(target_positions)
            dwell_counter = 0
            print(f"\nMoving to pose {current_target_idx + 1}/{len(target_positions)}")
        
        # Set the current target position
        robot.set_joint_position_target(target_positions[current_target_idx])
        
        # Write data to simulation
        robot.write_data_to_sim()
        
        # Perform simulation step
        sim.step()
        
        # Update robot state
        robot.update(sim_dt)
        
        # Increment counters
        dwell_counter += 1
        step_count += 1
    
    print("Simulation complete! You've worked with a robotic arm in Isaac Lab.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 