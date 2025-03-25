# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to implement a pick-and-place task with a robot arm.

This tutorial helps beginners understand:
1. How to create a complete pick-and-place application
2. How to use contact sensors to detect successful grasps
3. How to control a gripper and coordinate arm movements
4. How to structure a task as a state machine

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/10_pick_and_place.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create an argument parser
parser = argparse.ArgumentParser(description="Tutorial on implementing a pick-and-place task")
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
import enum
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, ArticulationCfg, RigidObjectCfg
from isaaclab.scene.ground import GroundPlaneCfg
from isaaclab.scene.table import TableCfg
from isaaclab.scene.light import DomeLightCfg
from isaaclab.controllers.ik.differential_ik import DifferentialIK, DifferentialIKCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sensors.contact import ContactSensor, ContactSensorCfg


# Define states for our pick-and-place state machine
class TaskState(enum.Enum):
    INIT = 0
    MOVE_TO_PREGRASP = 1
    MOVE_TO_GRASP = 2
    CLOSE_GRIPPER = 3
    LIFT_OBJECT = 4
    MOVE_TO_PLACE = 5
    PLACE_OBJECT = 6
    OPEN_GRIPPER = 7
    MOVE_TO_HOME = 8
    MOVE_TO_HOME_FINAL = 9  # Added this state for the final return to home
    TASK_COMPLETE = 10


def main():
    """Main function that demonstrates a pick-and-place task."""
    
    print("Welcome to Isaac Lab! Let's implement a pick-and-place task.")
    
    # Step 1: Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # Step 2: Set up a camera view appropriate for viewing the task
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.5])
    
    # Step 3: Create a scene
    scene = InteractiveScene(sim)
    
    # Step 4: Add a ground plane
    ground_cfg = GroundPlaneCfg(
        size=np.array([10.0, 10.0]),
        color=np.array([0.2, 0.2, 0.2, 1.0])
    )
    scene.add_ground(ground_cfg)
    
    # Step 5: Add lighting for better visualization
    light_cfg = DomeLightCfg(
        intensity=1000.0,
        color=np.array([1.0, 1.0, 1.0])
    )
    scene.add_dome_light(light_cfg)
    
    # Step 6: Add a table
    table_cfg = TableCfg(
        prim_path="/World/table",
        position=np.array([0.5, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=np.array([0.5, 0.7, 0.5])
    )
    table = scene.add_table(table_cfg)
    
    # Step 7: Add a Franka Panda robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/robot",
        name="panda_robot",
        usd_path="{ISAACLAB_ASSETS}/robots/franka_panda/panda.usd",
        init_state={
            "pos": np.array([0.0, 0.0, 0.0]),
            # Start with a neutral pose and open gripper
            "dof_pos": np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0, 0.04, 0.04])
        }
    )
    robot = scene.add_articulation(robot_cfg)
    
    # Step 8: Add an object to pick up
    # We'll use a small colored cube
    cube_cfg = RigidObjectCfg(
        prim_path="/World/target_cube",
        name="target_cube",
        shape_cfg={"type": "box", "size": np.array([0.04, 0.04, 0.04])},
        init_state={
            "pos": np.array([0.5, 0.0, 0.4 + 0.02])  # On top of the table
        },
        color=np.array([1.0, 0.0, 0.0, 1.0]),  # Red
    )
    target_cube = scene.add_rigid_object(cube_cfg)
    
    # Step 9: Add a target location marker for placing the object
    # Use a flat cylinder as the target location
    target_location_cfg = RigidObjectCfg(
        prim_path="/World/target_location",
        name="target_location",
        shape_cfg={"type": "cylinder", "radius": 0.05, "height": 0.001},
        init_state={
            "pos": np.array([0.5, 0.3, 0.4 + 0.0005])  # On top of the table, offset in Y
        },
        color=np.array([0.0, 1.0, 0.0, 1.0]),  # Green
    )
    target_location = scene.add_rigid_object(target_location_cfg)
    
    # Step 10: Set up the differential IK controller
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
    
    # Step 11: Add contact sensors to the gripper fingers
    contact_cfg = ContactSensorCfg(
        sensor_tick=0.0,  # Update every physics step
        history_size=1    # Keep only the most recent contact
    )
    
    # Left finger contact sensor
    left_finger_sensor = ContactSensor(
        prim_path=f"{robot.prim_path}/panda_leftfinger",
        name="left_finger_sensor",
        cfg=contact_cfg
    )
    left_finger_sensor.initialize()
    
    # Right finger contact sensor
    right_finger_sensor = ContactSensor(
        prim_path=f"{robot.prim_path}/panda_rightfinger",
        name="right_finger_sensor",
        cfg=contact_cfg
    )
    right_finger_sensor.initialize()
    
    # Step 12: Reset the scene to initialize physics
    scene.reset()
    print("[INFO]: Scene setup complete!")
    
    # Define key poses for the pick-and-place task
    # Home position - a safe, neutral position
    home_position = np.array([0.3, 0.0, 0.5])
    # Pre-grasp position - slightly above the object
    pre_grasp_position = np.array([0.5, 0.0, 0.5])
    # Grasp position - at the object's position but slightly higher to account for gripper
    grasp_position = np.array([0.5, 0.0, 0.45])
    # Post-grasp lift position - lifting the object
    lift_position = np.array([0.5, 0.0, 0.6])
    # Pre-place position - above the target location
    pre_place_position = np.array([0.5, 0.3, 0.6])
    # Place position - at the target location
    place_position = np.array([0.5, 0.3, 0.45])
    
    # Orientation for the gripper (pointing downward)
    downward_orientation = quat_from_euler_xyz(np.array([math.pi, 0.0, 0.0]))
    
    # Initialize task state and counters
    current_state = TaskState.INIT
    state_timer = 0
    
    # Function to control the gripper
    def set_gripper(gripper_width):
        joint_positions = robot.get_joint_positions()
        # The last two DOFs are the gripper fingers
        joint_positions[-2:] = [gripper_width, gripper_width]
        robot.set_joint_positions(joint_positions)
    
    # Function to check if gripper contacts the target cube
    def check_gripper_contact():
        left_contacts = left_finger_sensor.get_contact_prims()
        right_contacts = right_finger_sensor.get_contact_prims()
        
        target_prim_path = target_cube.prim_path
        
        if target_prim_path in left_contacts or target_prim_path in right_contacts:
            return True
        return False
    
    # Function to move the robot's end-effector to a target pose
    def move_to_target(target_position, target_orientation):
        # Apply IK to move towards the target
        joint_positions = ik_controller.compute_inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation
        )
        
        # Set joint positions for the arm (not the gripper)
        robot_joint_positions = robot.get_joint_positions()
        robot_joint_positions[:robot_dof] = joint_positions
        robot.set_joint_positions(robot_joint_positions)
    
    # Function to check if end-effector is close to a target position
    def is_at_target(target_position, tolerance=0.02):
        current_ee_pos, _ = robot.get_ee_transform()
        distance = np.linalg.norm(current_ee_pos - target_position)
        return distance < tolerance
    
    # Step 13: Run the simulation and execute the pick-and-place task
    step_count = 0
    
    while simulation_app.is_running() and step_count < 4000:  # Run for ~40 seconds or until complete
        # Execute the current state in the state machine
        if current_state == TaskState.INIT:
            print("Initializing task: Moving to home position")
            current_state = TaskState.MOVE_TO_HOME
        
        elif current_state == TaskState.MOVE_TO_HOME:
            # Move to the home position
            move_to_target(home_position, downward_orientation)
            
            # Check if we've reached the home position
            if is_at_target(home_position):
                print("Reached home position. Moving to pre-grasp position...")
                current_state = TaskState.MOVE_TO_PREGRASP
                state_timer = 0
        
        elif current_state == TaskState.MOVE_TO_PREGRASP:
            # Open the gripper wide
            set_gripper(0.04)
            
            # Move to position above the object
            move_to_target(pre_grasp_position, downward_orientation)
            
            # Check if we've reached the pre-grasp position
            if is_at_target(pre_grasp_position):
                print("Reached pre-grasp position. Moving to grasp position...")
                current_state = TaskState.MOVE_TO_GRASP
                state_timer = 0
        
        elif current_state == TaskState.MOVE_TO_GRASP:
            # Move down to the object
            move_to_target(grasp_position, downward_orientation)
            
            # Check if we've reached the grasp position
            if is_at_target(grasp_position):
                print("Reached grasp position. Closing gripper...")
                current_state = TaskState.CLOSE_GRIPPER
                state_timer = 0
        
        elif current_state == TaskState.CLOSE_GRIPPER:
            # Close the gripper
            set_gripper(0.0)
            
            # Wait for the gripper to close and check for contact
            state_timer += 1
            if state_timer > 50:  # Wait half a second
                if check_gripper_contact():
                    print("Object grasped successfully! Lifting object...")
                    current_state = TaskState.LIFT_OBJECT
                else:
                    print("Failed to grasp object. Returning to home...")
                    current_state = TaskState.MOVE_TO_HOME
                state_timer = 0
        
        elif current_state == TaskState.LIFT_OBJECT:
            # Lift the object
            move_to_target(lift_position, downward_orientation)
            
            # Check if we've lifted the object
            if is_at_target(lift_position):
                print("Object lifted. Moving to pre-place position...")
                current_state = TaskState.MOVE_TO_PLACE
                state_timer = 0
        
        elif current_state == TaskState.MOVE_TO_PLACE:
            # Move to position above the target location
            move_to_target(pre_place_position, downward_orientation)
            
            # Check if we've reached the pre-place position
            if is_at_target(pre_place_position):
                print("Reached pre-place position. Placing object...")
                current_state = TaskState.PLACE_OBJECT
                state_timer = 0
        
        elif current_state == TaskState.PLACE_OBJECT:
            # Move down to place the object
            move_to_target(place_position, downward_orientation)
            
            # Check if we've reached the place position
            if is_at_target(place_position):
                print("Reached place position. Opening gripper...")
                current_state = TaskState.OPEN_GRIPPER
                state_timer = 0
        
        elif current_state == TaskState.OPEN_GRIPPER:
            # Open the gripper to release the object
            set_gripper(0.04)
            
            # Wait for the gripper to open
            state_timer += 1
            if state_timer > 50:  # Wait half a second
                print("Object placed successfully! Returning to home...")
                # Use our new state for the final return home
                current_state = TaskState.MOVE_TO_HOME_FINAL
                state_timer = 0
        
        elif current_state == TaskState.MOVE_TO_HOME_FINAL:
            # Move to the home position for the final time
            move_to_target(home_position, downward_orientation)
            
            # Check if we've reached the home position
            if is_at_target(home_position):
                print("Returned home. Task completed!")
                current_state = TaskState.TASK_COMPLETE
                state_timer = 0
        
        elif current_state == TaskState.TASK_COMPLETE:
            print("Task completed successfully!")
            print("Simulation can now be ended.")
            break
        
        # Print current state periodically
        if step_count % 100 == 0:
            current_ee_pos, _ = robot.get_ee_transform()
            print(f"Step {step_count} - State: {current_state.name}")
            print(f"  End-effector position: {current_ee_pos}")
            
            # Check for and print contact information
            if check_gripper_contact():
                print("  Gripper is in contact with the target object")
        
        # Perform one simulation step
        scene.step()
        step_count += 1
    
    print("Simulation complete! You've implemented a pick-and-place task in Isaac Lab.")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation application when done
    simulation_app.close() 