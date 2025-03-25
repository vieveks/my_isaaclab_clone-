# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to add sensors to a robot and visualize camera images.

This tutorial extends the previous example by:
1. Adding a camera to the robot's end-effector
2. Adding contact sensors to the gripper
3. Displaying camera images in a separate window using matplotlib

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/05.5_visualize_camera.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import datetime

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on visualizing robot camera images")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Make sure cameras are enabled for this script
if not args_cli.enable_cameras:
    print("[WARNING] This script requires enabling cameras. Adding --enable_cameras flag automatically.")
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import threading
import time

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, patterns
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip


@configclass
class RobotSensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with a Franka Panda robot and sensors."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.copy().replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # Enable contact sensors on the robot
        spawn=FRANKA_PANDA_CFG.spawn.replace(activate_contact_sensors=True)
    )
    
    # Add a cube for the gripper to interact with
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.1),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )

    # camera attached to end-effector (hand)
    hand_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera",
        update_period=0.1,  # Update at 10 Hz
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.01, 1.0e3)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.08, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    
    # contact sensor on gripper fingers
    # Using regex pattern to match finger names
    finger_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger", 
        update_period=0.0,  # update every physics step
        history_length=6, 
        debug_vis=True
    )


class CameraVisualizer:
    """Class to visualize camera images by saving them to files."""
    
    def __init__(self, base_output_dir="camera_output"):
        """Initialize the visualizer with an output directory."""
        # Create a unique folder for this run using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for rgb and depth images
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.depth_dir = os.path.join(self.output_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        # Initialize with empty images
        self.rgb_img = np.zeros((240, 320, 3))
        self.depth_img = np.zeros((240, 320))
        
        # Initialize counter for saving images
        self.frame_count = 0
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        self.running = False
        
        print(f"Camera images will be saved to {os.path.abspath(self.output_dir)}")
        print(f"  RGB images: {os.path.abspath(self.rgb_dir)}")
        print(f"  Depth images: {os.path.abspath(self.depth_dir)}")
    
    def update_images(self, rgb_data, depth_data):
        """Update the stored images with new data and save them to files."""
        with self.lock:
            # Update RGB image
            if rgb_data is not None:
                # Convert from tensor to numpy array and normalize if needed
                if isinstance(rgb_data, torch.Tensor):
                    self.rgb_img = rgb_data.detach().cpu().numpy()
                else:
                    self.rgb_img = rgb_data
                
                # Remove batch dimension if present
                if len(self.rgb_img.shape) == 4:
                    self.rgb_img = self.rgb_img[0]  # Remove batch dimension
                
                # Ensure we're in [0, 1] range for saving
                if self.rgb_img.max() > 1.0:
                    self.rgb_img = self.rgb_img / 255.0
            
            # Update depth image
            if depth_data is not None:
                # Convert from tensor to numpy array
                if isinstance(depth_data, torch.Tensor):
                    self.depth_img = depth_data.detach().cpu().numpy()
                else:
                    self.depth_img = depth_data
                
                # Remove batch dimension if present
                if len(self.depth_img.shape) == 4:
                    self.depth_img = self.depth_img[0]  # Remove batch dimension
                
                # Handle multi-dimensional data (take the first channel if needed)
                if len(self.depth_img.shape) > 2:
                    self.depth_img = self.depth_img[..., 0]
            
            # Save images periodically (every 10 frames)
            if self.frame_count % 10 == 0:
                self._save_current_images()
            
            self.frame_count += 1
    
    def _save_current_images(self):
        """Save the current RGB and depth images to files."""
        # Save RGB image
        rgb_filename = os.path.join(self.rgb_dir, f"frame_{self.frame_count:04d}.png")
        plt.imsave(rgb_filename, self.rgb_img)
        
        # Save depth image with colormap
        depth_filename = os.path.join(self.depth_dir, f"frame_{self.frame_count:04d}.png")
        plt.imsave(depth_filename, self.depth_img, cmap='viridis')
        
        # Print update occasionally
        if self.frame_count % 100 == 0:
            print(f"Saved camera images at frame {self.frame_count}")
    
    def start_update_thread(self):
        """No need for an update thread when saving to files."""
        self.running = True
    
    def stop(self):
        """Stop the visualizer."""
        self.running = False
        print(f"Camera visualizer stopped. {self.frame_count} frames processed.")
        print(f"Images saved to {os.path.abspath(self.output_dir)}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator with the sensors and visualize camera images."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    step_count = 0
    
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
    
    # Add an additional position for interacting with the cube
    grab_cube_position = torch.tensor([0.5, 0.0, 0.0, -1.57, 0.0, 1.57, 0.8, 0.01, 0.01], device=sim.device)
    target_positions.append(grab_cube_position)
    
    current_target_idx = 0
    dwell_steps = 200  # Stay at each position for 2 seconds
    dwell_counter = 0
    
    # Initialize fingers before the first reset
    try:
        print("Initializing contact sensors on fingers...")
        # Check if sensors are initialized
        if hasattr(scene["finger_contacts"], "is_initialized"):
            print(f"Contact sensors initialized: {scene['finger_contacts'].is_initialized}")
    except KeyError:
        print("No finger contact sensors found in scene")
    
    # Create camera visualizer
    visualizer = CameraVisualizer()
    visualizer.start_update_thread()
    
    # Simulate physics
    try:
        while simulation_app.is_running() and step_count < 2000:  # Run for ~20 seconds
            # Reset every 500 steps
            if step_count % 500 == 0:
                # Reset the robot position
                root_state = scene["robot"].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene["robot"].write_root_pose_to_sim(root_state[:, :7])
                scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
                
                # Reset joint positions
                joint_pos = scene["robot"].data.default_joint_pos.clone()
                joint_vel = scene["robot"].data.default_joint_vel.clone()
                scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
                
                # Reset cube position
                try:
                    scene["cube"].reset()
                except KeyError:
                    print("Cube not found in scene")
                
                # Clear internal buffers - be careful with this
                try:
                    scene.reset()
                    print("[INFO]: Resetting robot state...")
                except Exception as e:
                    print(f"Warning during reset: {e}")
                    # Continue even if reset fails
                
                # Reset counters
                dwell_counter = 0
                current_target_idx = 0
            
            # Update camera visualization
            try:
                if scene["hand_camera"].data.output is not None:
                    rgb_data = scene["hand_camera"].data.output["rgb"]
                    depth_data = scene["hand_camera"].data.output["distance_to_image_plane"]
                    visualizer.update_images(rgb_data, depth_data)
            except (KeyError, AttributeError) as e:
                print(f"Error updating camera visualization: {e}")
            
            # Print sensor data periodically
            if step_count % 50 == 0:
                print("\n----------- Step", step_count, "------------")
                
                # Print camera info if available
                try:
                    if scene["hand_camera"].data.output is not None:
                        print("Camera on end-effector:")
                        rgb_shape = scene["hand_camera"].data.output["rgb"].shape
                        depth_shape = scene["hand_camera"].data.output["distance_to_image_plane"].shape
                        print(f"  RGB image shape: {rgb_shape}")
                        print(f"  Depth image shape: {depth_shape}")
                except (KeyError, AttributeError):
                    pass
                
                # Print contact sensor info if available
                try:
                    contact_forces = scene["finger_contacts"].data.net_forces_w
                    if contact_forces is not None and contact_forces.numel() > 0:
                        max_force = torch.max(contact_forces).item()
                        print(f"Gripper contact force: {max_force:.4f}")
                    else:
                        print("Gripper contact force: No contact detected")
                except (KeyError, AttributeError):
                    pass
            
            # Determine if we need to change targets
            if dwell_counter >= dwell_steps:
                current_target_idx = (current_target_idx + 1) % len(target_positions)
                dwell_counter = 0
                print(f"\nMoving to pose {current_target_idx + 1}/{len(target_positions)}")
            
            # Set the current target position
            scene["robot"].set_joint_position_target(target_positions[current_target_idx])
            
            # Write data to simulation
            scene.write_data_to_sim()
            
            # Perform simulation step
            sim.step()
            
            # Update scene and sensors
            scene.update(sim_dt)
            
            # Increment counters
            dwell_counter += 1
            step_count += 1
    finally:
        # Ensure visualizer is properly stopped
        visualizer.stop()


def main():
    """Main function."""
    
    print("Welcome to Isaac Lab! Let's visualize robot camera images.")
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set up a camera view appropriate for viewing the robot
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    
    # Create the scene with robot and sensors
    scene_cfg = RobotSensorsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset the simulation
    sim.reset()
    
    print("[INFO]: Simulation setup complete with robot arm and sensors!")
    print("[INFO]: Added a camera to the end-effector and contact sensors to the gripper fingers")
    print("[INFO]: Camera images will be saved to files")
    
    # Run the simulator
    run_simulator(sim, scene)
    
    print("Simulation complete! You've visualized camera images from a robot in Isaac Lab.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 