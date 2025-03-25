# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to work with the G1 humanoid robot's camera and use YOLOv8 for object detection.

In this tutorial, you'll learn how to:
1. Set up a scene with a G1 humanoid robot and camera
2. Capture imagery from the robot's perspective
3. Process images with YOLOv8 for real-time object detection
4. Visualize detection results overlaid on camera feed

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/05.6_g1_camera_yolov8.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import datetime
from typing import Optional, Dict, Any

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on YOLOv8 object detection with G1 robot camera")
parser.add_argument(
    "--save",
    action="store_true",
    default=True,
    help="Save the camera data and detection results to disk.",
)
parser.add_argument(
    "--model",
    type=str,
    default="yolov8n.pt",
    help="YOLOv8 model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
)
parser.add_argument(
    "--confidence",
    type=float,
    default=0.5,
    help="Confidence threshold for YOLOv8 detections",
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
import cv2

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

# Import YOLOv8 for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print(f"[INFO]: YOLOv8 detected. Will use {args_cli.model} for object detection.")
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING]: YOLOv8 not found. Please install with 'pip install ultralytics' to enable object detection.")
    print("[WARNING]: Continuing without object detection capability.")

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
    # Create camera prim at the head position
    prim_utils.create_prim("/World/Robot/head_link/head_camera", "Xform")
    
    # Configure camera - increase resolution to meet minimum requirement of 300 pixels
    camera_cfg = CameraCfg(
        prim_path="/World/Robot/head_link/head_camera/CameraSensor",
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
            pos=(0.15, 0.0, 0.1),        # Moved slightly more forward
            rot=(0.2617994, 0.0, 0.0, 0.9659258),  # 30-degree tilt downward (pi/6 radians)
            convention="ros"              # Using ROS convention
        )
    )
    
    # Create camera
    camera = Camera(cfg=camera_cfg)
    print(f"[INFO]: Camera attached to robot at position (0.15, 0.0, 0.1) relative to head_link")
    
    # Create a viewport window showing the camera view using Replicator
    try:
        camera_path = camera_cfg.prim_path
        rep.create.camera_viewport(camera_path=camera_path, viewport_name="Robot Camera View")
        print("[INFO]: Camera viewport window created. Look for 'Robot Camera View' tab in the interface.")
    except Exception as e:
        print(f"[WARNING]: Could not create camera viewport window: {e}")
    
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
    
    # Create a variety of objects for YOLOv8 to detect - add common COCO dataset classes
    # Add a person (represented by a cylinder)
    person_cfg = sim_utils.CylinderCfg(
        radius=0.2,
        height=1.8,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.6)),
    )
    person_cfg.func("/World/Person", person_cfg, translation=(2.5, -1.0, 0.9))
    print(f"[INFO]: Added a person representation at position (2.5, -1.0, 0.9)")
    
    # Add a car (represented by a cuboid)
    car_cfg = sim_utils.CuboidCfg(
        size=(2.0, 1.0, 0.8),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
    )
    car_cfg.func("/World/Car", car_cfg, translation=(4.0, 1.0, 0.4))
    print(f"[INFO]: Added a car representation at position (4.0, 1.0, 0.4)")
    
    # Add a chair (represented by a complex shape)
    seat_cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    seat_cfg.func("/World/Chair/Seat", seat_cfg, translation=(1.5, 1.5, 0.5))
    
    backrest_cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.1, 0.6),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    backrest_cfg.func("/World/Chair/Back", backrest_cfg, translation=(1.5, 1.25, 0.8))
    print(f"[INFO]: Added a chair representation at position (1.5, 1.5, 0.5)")
    
    # Add a traffic light (represented by a torus)
    light_box_cfg = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.6),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
    )
    light_box_cfg.func("/World/TrafficLight/Box", light_box_cfg, translation=(3.0, 0.0, 2.5))
    
    red_light_cfg = sim_utils.SphereCfg(
        radius=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    red_light_cfg.func("/World/TrafficLight/Red", red_light_cfg, translation=(3.0, 0.0, 2.7))
    
    yellow_light_cfg = sim_utils.SphereCfg(
        radius=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
    )
    yellow_light_cfg.func("/World/TrafficLight/Yellow", yellow_light_cfg, translation=(3.0, 0.0, 2.5))
    
    green_light_cfg = sim_utils.SphereCfg(
        radius=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    green_light_cfg.func("/World/TrafficLight/Green", green_light_cfg, translation=(3.0, 0.0, 2.3))
    print(f"[INFO]: Added a traffic light representation at position (3.0, 0.0, 2.5)")
    
    # Add a red cube - closer to the camera for easy detection
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
    
    return scene_entities


class YOLODetector:
    """Class to handle YOLOv8 object detection."""
    
    def __init__(self, model_name="yolov8n.pt", confidence=0.5):
        """Initialize the YOLOv8 detector."""
        self.available = YOLO_AVAILABLE
        if not self.available:
            return
        
        try:
            self.model = YOLO(model_name)
            print(f"[INFO]: YOLOv8 model '{model_name}' loaded successfully!")
        except Exception as e:
            print(f"[ERROR]: Failed to load YOLOv8 model: {e}")
            self.available = False
            return
        
        self.confidence = confidence
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        print(f"[INFO]: YOLOv8 detector initialized with confidence threshold {confidence}")
    
    def detect(self, image):
        """Run object detection on an image and return results."""
        if not self.available:
            return None
        
        try:
            # Convert to BGR format for OpenCV if needed
            if isinstance(image, torch.Tensor):
                image_np = image.detach().cpu().numpy()
                # Remove batch dimension if present
                if len(image_np.shape) == 4:
                    image_np = image_np[0]
                # Convert from [0,1] to [0,255]
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
            
            # Run detection with confidence threshold
            results = self.model(image_np, conf=self.confidence)
            return results
        except Exception as e:
            print(f"[ERROR]: YOLOv8 detection failed: {e}")
            return None
    
    def draw_detections(self, image, results):
        """Draw detection boxes and labels on the image."""
        if not self.available or results is None:
            return image
        
        try:
            # Get the result object
            result = results[0] if isinstance(results, list) else results
            
            # Convert image to correct format
            if isinstance(image, torch.Tensor):
                image_np = image.detach().cpu().numpy()
                # Remove batch dimension if present
                if len(image_np.shape) == 4:
                    image_np = image_np[0]
                # Convert from [0,1] to [0,255]
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image.copy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
            
            # Draw boxes and labels
            image_with_boxes = result.plot()
            
            # Convert back to [0,1] range if the original was in that range
            if image.max() <= 1.0:
                image_with_boxes = image_with_boxes.astype(np.float32) / 255.0
            
            return image_with_boxes
        except Exception as e:
            print(f"[ERROR]: Drawing detections failed: {e}")
            return image


class CameraVisualizer:
    """Class to visualize camera images in real-time and save them to files."""
    
    def __init__(self, base_output_dir="camera_output", yolo_detector: Optional[YOLODetector] = None):
        """Initialize the visualizer with an output directory."""
        # Store the YOLOv8 detector
        self.yolo_detector = yolo_detector
        
        # Create a unique folder for this run using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_output_dir, f"g1_camera_yolo_{timestamp}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for rgb and depth images
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.depth_dir = os.path.join(self.output_dir, "depth")
        self.detection_dir = os.path.join(self.output_dir, "detection")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.detection_dir, exist_ok=True)
        
        # Initialize with empty images - updated for larger resolution
        self.rgb_img = np.zeros((320, 480, 3))
        self.depth_img = np.zeros((320, 480))
        self.detection_img = np.zeros((320, 480, 3))
        
        # Store latest detection results
        self.detection_results = None
        
        # Initialize counter for saving images
        self.frame_count = 0
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        self.running = False
        
        # Create matplotlib figure for visualization
        self.fig = Figure(figsize=(18, 6))
        self.canvas = FigureCanvas(self.fig)
        
        # Create subplots for RGB, depth images, and detections
        self.ax_rgb = self.fig.add_subplot(131)
        self.ax_depth = self.fig.add_subplot(132)
        self.ax_detection = self.fig.add_subplot(133)
        
        # Initialize image display with new dimensions
        self.rgb_plot = self.ax_rgb.imshow(np.zeros((320, 480, 3)))
        self.depth_plot = self.ax_depth.imshow(np.zeros((320, 480)), cmap='viridis')
        self.detection_plot = self.ax_detection.imshow(np.zeros((320, 480, 3)))
        
        # Set titles
        self.ax_rgb.set_title('RGB Camera View')
        self.ax_depth.set_title('Depth View')
        self.ax_detection.set_title('YOLOv8 Detections')
        
        print(f"Camera and detection images will be saved to {os.path.abspath(self.output_dir)}")
        print(f"  RGB images: {os.path.abspath(self.rgb_dir)}")
        print(f"  Depth images: {os.path.abspath(self.depth_dir)}")
        print(f"  Detection images: {os.path.abspath(self.detection_dir)}")
    
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
                
                # Run object detection if YOLOv8 is available
                if self.yolo_detector is not None and self.yolo_detector.available:
                    self.detection_results = self.yolo_detector.detect(self.rgb_img)
                    self.detection_img = self.yolo_detector.draw_detections(self.rgb_img, self.detection_results)
            
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
        """Save the current RGB, depth, and detection images to files."""
        # Save RGB image
        rgb_filename = os.path.join(self.rgb_dir, f"frame_{self.frame_count:04d}.png")
        plt.imsave(rgb_filename, self.rgb_img)
        
        # Save depth image with colormap
        depth_filename = os.path.join(self.depth_dir, f"frame_{self.frame_count:04d}.png")
        plt.imsave(depth_filename, self.depth_img, cmap='viridis')
        
        # Save detection image
        if self.yolo_detector is not None and self.yolo_detector.available:
            detection_filename = os.path.join(self.detection_dir, f"frame_{self.frame_count:04d}.png")
            plt.imsave(detection_filename, self.detection_img)
        
        # Print update occasionally
        if self.frame_count % 100 == 0:
            print(f"Saved camera images at frame {self.frame_count}")
    
    def _update_visualization(self):
        """Update the matplotlib visualization."""
        self.rgb_plot.set_data(self.rgb_img)
        self.depth_plot.set_data(self.depth_img)
        
        if self.yolo_detector is not None and self.yolo_detector.available:
            self.detection_plot.set_data(self.detection_img)
        
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
        print(f"Visualizer stopped. {self.frame_count} frames processed.")
        print(f"Images saved to {os.path.abspath(self.output_dir)}")


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict, yolo_detector: Optional[YOLODetector] = None):
    """Run the simulator with the G1 robot and camera."""
    # Extract camera and robot from scene entities
    camera: Camera = scene_entities["camera"]
    robot = scene_entities.get("robot", None)
    
    # Create replicator writer if saving is enabled
    if args_cli.save:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera_yolo")
        rep_writer = rep.BasicWriter(
            output_dir=output_dir,
            frame_padding=0,
            colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
        )
    
    # Create camera visualizer with YOLOv8 detector
    visualizer = CameraVisualizer(yolo_detector=yolo_detector)
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
    
    # Create rotation to occasionally view different objects
    rotation_period = 10.0  # seconds per rotation
    rotation_angle = 0.0
    
    try:
        print("\n[INFO]: Starting simulation with G1 humanoid robot camera and YOLOv8 detection...")
        while simulation_app.is_running() and sim_time < 60.0:  # Run for 60 seconds
            # Process camera data if available
            try:
                # Access camera data via data.output attribute
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
            
            # Rotate the robot occasionally to view different objects
            if robot is not None and step_count % 100 == 0:
                # Calculate new angle
                rotation_angle = (sim_time / rotation_period) * 2 * np.pi
                
                # Get current position of the robot
                pos = robot.data.root_pos_w.clone()[0]
                
                # Create rotation quaternion (around Z-axis)
                quat = torch.tensor([np.cos(rotation_angle/2), 0, 0, np.sin(rotation_angle/2)], device=robot.data.root_pos_w.device)
                
                # Apply new position and rotation
                robot.write_root_pose_to_sim(torch.cat([pos.unsqueeze(0), quat.unsqueeze(0)], dim=1))
            
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
                if YOLO_AVAILABLE and visualizer.detection_results is not None:
                    # Get detection results
                    result = visualizer.detection_results[0] if isinstance(visualizer.detection_results, list) else visualizer.detection_results
                    # Print detected classes and confidence
                    print("Detected objects:")
                    detected_classes = {}
                    for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                        class_name = yolo_detector.class_names[int(cls)]
                        if class_name not in detected_classes:
                            detected_classes[class_name] = []
                        detected_classes[class_name].append(float(conf))
                    
                    for class_name, confs in detected_classes.items():
                        avg_conf = sum(confs) / len(confs)
                        print(f"  - {class_name}: {len(confs)} instances (avg conf: {avg_conf:.2f})")
    
    finally:
        # Ensure proper cleanup
        visualizer.stop()
        if args_cli.save:
            # BasicWriter doesn't have a close method, so we don't need to call it
            pass


def main():
    """Main function."""
    
    print("Welcome to the G1 Humanoid Robot Camera Tutorial with YOLOv8 Object Detection!")
    
    # Initialize YOLOv8 detector if available
    yolo_detector = None
    if YOLO_AVAILABLE:
        yolo_detector = YOLODetector(model_name=args_cli.model, confidence=args_cli.confidence)
    
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
    run_simulator(sim, scene_entities, yolo_detector)
    
    print("Simulation complete! You've worked with the G1 humanoid robot's camera and YOLOv8 object detection in Isaac Lab.")


if __name__ == "__main__":
    # Run the main function
    main()
    
    # Close simulation app
    simulation_app.close() 