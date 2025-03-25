# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add a camera to the G1 robot's head and visualize the RGB data.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p g1_head_camera.py --enable_cameras

"""

###############################
# PART 1: SETUP AND INITIALIZATION
###############################

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="G1 robot with head camera visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Make sure cameras are enabled for this script
# This is critical - the simulation won't display camera data without this flag
if not args_cli.enable_cameras:
    print("[WARNING] This script requires enabling cameras. Adding --enable_cameras flag automatically.")
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Import G1 configuration
from isaaclab_assets import G1_MINIMAL_CFG


###############################
# PART 2: SCENE CONFIGURATION
###############################

@configclass
class G1CameraSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with G1 robot and head camera.
    
    This class defines all the components that will be added to our simulation scene:
    - Ground plane for the robot to stand on
    - Lighting for visibility
    - The G1 robot
    - A camera attached to the robot's head
    """

    # Add a ground plane to the scene
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Add dome light for better visibility
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Configure the G1 robot
    # {ENV_REGEX_NS} is a placeholder that will be replaced with the actual environment namespace
    robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Add a head camera to the robot
    # The camera is attached to the G1's head link (in this case, using the torso_link)
    head_camera = CameraCfg(
        # Attach the camera to the torso_link of the robot
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
        update_period=0.1,  # Update every 100ms (10Hz)
        height=240,         # Lower resolution for faster processing
        width=320,
        data_types=["rgb"],  # Just RGB data for this example
        
        # Configure the camera parameters
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,          # Controls the field of view
            focus_distance=400.0,       # Distance at which objects are in focus
            horizontal_aperture=20.955, # Affects the field of view
            clipping_range=(0.1, 1.0e5) # Min and max view distances
        ),
        
        # Offset the camera position to simulate being on the "head"
        # These values position the camera forward and up from the torso link
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 0.2),          # Forward and up from the torso
            rot=(0.0, 0.0, 0.0, 1.0),     # Default orientation (facing forward)
            convention="ros"               # Using ROS coordinate convention
        ),
    )


###############################
# PART 3: SIMULATION LOOP
###############################

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop.
    
    This function:
    1. Creates a simple animation for the robot
    2. Captures and processes camera data
    3. Analyzes the image to identify what the camera is looking at
    """
    # Extract scene entities for easier access
    robot = scene["robot"]
    head_camera = scene["head_camera"]

    # Define simulation stepping parameters
    sim_dt = sim.get_physics_dt()  # Physics time step
    count = 0                      # Step counter
    
    # Create a simple walking pattern for demonstration
    # Initialize joint position targets
    joint_pos_targets = torch.zeros((scene.num_envs, robot.num_joints), device=sim.device)
    
    # Set the robot to a stable standing position (use default)
    joint_pos_targets[:] = robot.data.default_joint_pos.clone()
    
    # Simulation loop - runs until the application is closed
    while simulation_app.is_running():
        # Reset occasionally to test camera operations
        if count % 500 == 0:
            count = 0
            # Reset the robot to default pose
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone()
            )
            # Clear internal buffers
            scene.reset()
            print("\n[INFO]: Resetting robot state...")
        
        # Create a simple animation by moving the torso joint
        # This makes the robot look around, which changes what the camera sees
        if count % 50 == 0:
            # Find the torso joint index
            torso_joint_idx = robot.find_joints("torso_joint")[0]
            
            # Create a sinusoidal motion for smooth oscillation
            angle = 0.1 * torch.sin(torch.tensor(count / 50.0))
            
            # Apply the angle to the torso joint
            joint_pos_targets[:, torso_joint_idx] = angle
        
        # Apply joint position targets to the robot
        robot.set_joint_position_target(joint_pos_targets)
        
        # Write data to simulation
        scene.write_data_to_sim()
        
        # Perform physics step
        sim.step()
        
        # Update scene and incremental counter
        scene.update(sim_dt)
        count += 1
        
        ###############################
        # PART 4: CAMERA DATA PROCESSING
        ###############################
        
        # Get and display camera data every 10 steps (to avoid flooding the terminal)
        if count % 10 == 0 and "rgb" in head_camera.data.output:
            # Get RGB data from the first environment
            # Camera data is a tensor of shape [num_envs, height, width, channels]
            rgb_data = head_camera.data.output["rgb"][0]
            
            # Print info about the RGB data
            print(f"\n[Camera Frame {count}] RGB shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
            
            # Convert to float for calculations (normalize to [0,1] range)
            rgb_data_float = rgb_data.float() / 255.0
            
            # Calculate and print some simple statistics about the image
            rgb_mean = torch.mean(rgb_data_float, dim=(0, 1)).cpu().numpy()
            rgb_min = torch.min(rgb_data_float).item()
            rgb_max = torch.max(rgb_data_float).item()
            
            print(f"RGB color averages (R,G,B): ({rgb_mean[0]:.3f}, {rgb_mean[1]:.3f}, {rgb_mean[2]:.3f})")
            print(f"RGB value range: [{rgb_min:.3f}, {rgb_max:.3f}]")
            
            # Simple scene classification based on color composition
            if rgb_mean[2] > 0.6:  # Lots of blue - likely looking at sky
                print("Camera likely viewing the sky")
            elif rgb_mean[1] > 0.5 and rgb_mean[0] < 0.3:  # Green dominant - grass or ground
                print("Camera likely viewing grass/ground")
            elif np.std(rgb_mean) < 0.1:  # All colors similar - maybe a wall or gray surface
                print("Camera likely viewing a uniform surface (wall/floor)")
            else:
                print("Camera viewing a mixed scene")


###############################
# PART 5: MAIN EXECUTION
###############################

def main():
    """Main function.
    
    Sets up the simulation, initializes the scene with the G1 robot and camera,
    and runs the simulation loop.
    """
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera to view the robot from a good angle
    # This is the viewpoint camera, not the robot's camera
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.5])
    
    # Create the scene with the configuration we defined
    scene_cfg = G1CameraSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset the simulation to initialize everything
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: G1 robot initialized with head camera. RGB data will be displayed in the terminal.")
    
    # Run the simulation loop
    run_simulator(sim, scene)


# Entry point of the script
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()