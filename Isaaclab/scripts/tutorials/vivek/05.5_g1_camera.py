# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to work with the G1 humanoid robot's camera in IsaacLab.

In this tutorial, you'll learn how to:
1. Set up a scene with a G1 humanoid robot
2. Add and configure the head camera
3. Save and visualize camera imagery from the robot's perspective

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/05.5_g1_camera.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import datetime

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on working with the G1 humanoid robot's camera")
parser.add_argument(
    "--save",
    action="store_true",
    default=True,
    help="Save the camera data to disk.",
)
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

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import threading
import time

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import configclass

# Import G1 velocity configuration
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

##
# Pre-defined configs - Import G1 robot configuration
##
try:
    # Use the flat environment configuration for visualization
    G1_ROBOT_CFG = G1FlatEnvCfg().scene.robot
    G1_AVAILABLE = True
except ImportError:
    print("[WARNING] Unitree G1 config not found. Using a placeholder configuration.")
    G1_ROBOT_CFG = None
    G1_AVAILABLE = False


def define_camera() -> Camera:
    """Define the camera sensor for the G1 robot."""
    # Create camera prim
    prim_utils.create_prim("/World/Robot/torso_link/head_camera", "Xform")
    
    # Configure camera - increase resolution to meet minimum requirement of 300 pixels
    camera_cfg = CameraCfg(
        prim_path="/World/Robot/torso_link/head_camera/CameraSensor",
        update_period=0.1,  # Update every 100ms (10Hz)
        height=320,  # Increased from 240
        width=480,   # Increased from 320
        data_types=["rgb", "distance_to_image_plane"],
        colorize_semantic_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.2),         # Moved forward more (0.2 -> 0.3) from the torso
            rot=(0.0, 0.0, 0.0, 1.0),    # Default orientation
            convention="ros"              # Using ROS convention
        ),
    )
    
    # Create camera
    camera = Camera(cfg=camera_cfg)
    print(f"[INFO]: Camera attached to robot at position (0.3, 0.0, 0.2) relative to torso_link")
    return camera


def design_scene() -> dict:
    """Design the scene with a G1 humanoid robot and camera."""
    # Create a dictionary for scene entities
    scene_entities = {}
    
    # Ground plane with better collision properties
    cfg = sim_utils.GroundPlaneCfg(
        size=(20.0, 20.0),
        color=(0.3, 0.3, 0.3)
    )
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Physics is enabled by default in this scene
    
    # Lighting
    cfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.8, 0.8, 0.8)
    )
    cfg.func("/World/Light", cfg)
    
    # Set up G1 robot
    if G1_AVAILABLE and G1_ROBOT_CFG is not None:
        robot_cfg = G1_ROBOT_CFG.copy().replace(
            prim_path="/World/Robot",
        )
        # Create articulation from config
        from isaaclab.assets import Articulation
        robot = Articulation(cfg=robot_cfg)
        scene_entities["robot"] = robot
    else:
        # Use a substitute robot if G1 is not available
        from isaaclab.assets import Articulation
        from isaaclab_assets.robots.humanoid import HUMANOID_CFG
        robot_cfg = HUMANOID_CFG.replace(
            prim_path="/World/Robot",
        )
        robot = Articulation(cfg=robot_cfg)
        scene_entities["robot"] = robot
    
    # Add camera
    camera = define_camera()
    scene_entities["camera"] = camera
    
    # Add some obstacles with better physics properties - make it brightly colored
    obstacle_cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.1,  # Added some damping
            angular_damping=0.1,  # Added some damping
            max_linear_velocity=100.0,  # Reduced from 1000.0
            max_angular_velocity=100.0,  # Reduced from 1000.0
            max_depenetration_velocity=10.0,  # Increased from 1.0
            solver_position_iteration_count=8,  # Added solver iterations
            solver_velocity_iteration_count=2,  # Added solver iterations
            enable_gyroscopic_forces=True,  # Enable stabilizing forces
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,
            rest_offset=0.01,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),  # Bright red for visibility
    )
    
    # Position obstacle directly in front of the robot at eye level
    obstacle_cfg.func("/World/Obstacle1", obstacle_cfg, translation=(1.5, 0.0, 1.0))
    print(f"[INFO]: Placed a bright red cube at position (1.5, 0.0, 1.0) - directly in front of the robot")
    
    # Add a second obstacle for more visual interest - blue
    obstacle2_cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),  # Smaller cube
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,
            rest_offset=0.01,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 1.0)),  # Blue
    )
    
    # Position the second obstacle a bit to the side
    obstacle2_cfg.func("/World/Obstacle2", obstacle2_cfg, translation=(1.7, 0.5, 0.8))
    print(f"[INFO]: Placed a blue cube at position (1.7, 0.5, 0.8) - to the right side of the view")
    
    return scene_entities


class CameraVisualizer:
    """Class to visualize camera images in real-time and save them to files."""
    
    def __init__(self, base_output_dir="camera_output"):
        """Initialize the visualizer with an output directory."""
        # Create a unique folder for this run using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_output_dir, f"g1_camera_{timestamp}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for rgb and depth images
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.depth_dir = os.path.join(self.output_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        # Initialize with empty images - updated for larger resolution
        self.rgb_img = np.zeros((320, 480, 3))
        self.depth_img = np.zeros((320, 480))
        
        # Initialize counter for saving images
        self.frame_count = 0
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        self.running = False
        
        # Create matplotlib figure for visualization
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        
        # Create subplots for RGB and depth images
        self.ax_rgb = self.fig.add_subplot(121)
        self.ax_depth = self.fig.add_subplot(122)
        
        # Initialize image display with new dimensions
        self.rgb_plot = self.ax_rgb.imshow(np.zeros((320, 480, 3)))
        self.depth_plot = self.ax_depth.imshow(np.zeros((320, 480)), cmap='viridis')
        
        # Set titles
        self.ax_rgb.set_title('RGB Camera View')
        self.ax_depth.set_title('Depth View')
        
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
            if self.frame_count % 10 == 0 and args_cli.save:
                self._save_current_images()
            
            # Update visualization
            if self.running:
                self._update_visualization()
            
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
    
    def _update_visualization(self):
        """Update the matplotlib visualization."""
        self.rgb_plot.set_data(self.rgb_img)
        self.depth_plot.set_data(self.depth_img)
        self.canvas.draw()
        plt.pause(0.001)  # Small pause to allow the GUI to update
    
    def start_visualization(self):
        """Start the visualization thread."""
        self.running = True
        plt.show(block=False)
    
    def stop(self):
        """Stop the visualizer."""
        self.running = False
        plt.close('all')
        print(f"Camera visualizer stopped. {self.frame_count} frames processed.")
        print(f"Images saved to {os.path.abspath(self.output_dir)}")


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator with the G1 robot and camera."""
    # Extract camera and robot from scene entities
    camera: Camera = scene_entities["camera"]
    robot = scene_entities.get("robot", None)
    
    # Create replicator writer if saving is enabled
    if args_cli.save:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
        rep_writer = rep.BasicWriter(
            output_dir=output_dir,
            frame_padding=0,
            colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
        )
    
    # Create camera visualizer
    visualizer = CameraVisualizer()
    visualizer.start_visualization()
    
    # Set initial robot pose if robot is available
    if G1_AVAILABLE and robot is not None:
        # Set initial joint positions for visualization
        robot_joints = {
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        }
        
        # Get default joint positions and apply our custom positions
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        
        # Update joint positions based on our configuration
        for pattern, value in robot_joints.items():
            # Find matching joints
            import re
            for i, name in enumerate(robot.data.joint_names):
                if re.match(pattern, name):
                    joint_pos[:, i] = value
        
        # Apply the joint positions
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Simulation loop
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    
    try:
        print("\n[INFO]: Starting simulation with G1 humanoid robot camera...")
        while simulation_app.is_running() and sim_time < 30.0:  # Run for 30 seconds
            # Process camera data if available
            try:
                # Access camera data via data.output attribute instead of get_camera_data() method
                if hasattr(camera, 'data') and hasattr(camera.data, 'output') and camera.data.output is not None:
                    rgb_data = camera.data.output.get("rgb")
                    depth_data = camera.data.output.get("distance_to_image_plane")
                    
                    if rgb_data is not None and depth_data is not None:
                        visualizer.update_images(rgb_data, depth_data)
                        
                        # Save data using replicator if enabled
                        if args_cli.save:
                            rep_writer.write(camera.data.output)
            except Exception as e:
                if step_count % 100 == 0:
                    print(f"Warning processing camera data: {e}")
            
            # Write robot data to sim if available
            if robot is not None:
                robot.write_data_to_sim()
            
            # Perform simulation step
            sim.step()
            
            # Update robot state if available
            if robot is not None:
                robot.update(sim_dt)
            
            # Update camera state
            camera.update(sim_dt)
            
            # Update time
            sim_time += sim_dt
            step_count += 1
            
            # Print progress periodically
            if step_count % 100 == 0:
                print(f"\n--- Step {step_count} (Time: {sim_time:.2f}s) ---")
    
    finally:
        # Ensure proper cleanup
        visualizer.stop()
        if args_cli.save:
            rep_writer.close()


def main():
    """Main function."""
    
    print("Welcome to the G1 Humanoid Robot Camera Tutorial!")
    
    # Check if G1 robot config is available
    if G1_AVAILABLE:
        print("[INFO]: Unitree G1 configuration found.")
    else:
        print("[INFO]: Unitree G1 configuration not found. Using a generic humanoid as a substitute.")
    
    # Initialize the simulation context with better physics settings
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,  # Match velocity config
        device="cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Physics is enabled by default in SimulationContext
    
    # Set up a camera view appropriate for viewing the robot and the scene
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.75])
    print("[INFO]: Main viewport camera positioned to view both the robot and the obstacles")
    
    # Design scene and get scene entities
    scene_entities = design_scene()
    
    # Reset the simulation
    sim.reset()
    
    print("[INFO]: G1 humanoid robot scene setup complete!")
    print("[INFO]: Robot equipped with head camera")
    if args_cli.save:
        print("[INFO]: Camera images will be saved to files")
    
    # Run the simulator
    run_simulator(sim, scene_entities)
    
    print("Simulation complete! You've worked with the G1 humanoid robot's camera in Isaac Lab.")


if __name__ == "__main__":
    # Run the main function
    main()
    
    # Close simulation app
    simulation_app.close() 