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
import numpy as np
import torch  # Ensure torch is imported at the top level
import threading
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
    default="yolov8s.pt",  # Using the small model instead of nano for better accuracy
    help="YOLOv8 model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
)
parser.add_argument(
    "--confidence",
    type=float,
    default=0.4,  # Slightly lower threshold to detect more objects
    help="Confidence threshold for YOLOv8 detections",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Device to run YOLOv8 on (cuda:0, cpu, etc.)",
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
    print("[INFO]: To install YOLOv8, run: 'pip install ultralytics'")

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
    
    # Configure camera - improved resolution and position for better detection
    camera_cfg = CameraCfg(
        prim_path="/World/Robot/head_link/head_camera/CameraSensor",
        update_period=0.05,  # Update every 50ms (20Hz) for more responsive detection
        height=640,  # Increased resolution
        width=640,   # Square format works better with YOLOv8
        data_types=["rgb", "distance_to_image_plane"],
        colorize_semantic_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 0.05),      # Moved forward and aligned better with eye level
            rot=(0.0, 0.0, 0.0, 1.0),  # No tilt - camera looks straight ahead
            convention="ros"           # Using ROS convention
        )
    )
    
    # Create camera
    camera = Camera(cfg=camera_cfg)
    print(f"[INFO]: Camera attached to robot head_link at position (0.2, 0.0, 0.05) looking straight ahead")
    
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
    
    # Lighting - improved for better object visibility
    cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,  # Increased intensity
        color=(1.0, 1.0, 1.0)  # Pure white for better color reproduction
    )
    cfg.func("/World/Light", cfg)
    
    # Set up G1 robot - positioned at a better location for viewing objects
    if G1_AVAILABLE and G1_ROBOT_CFG is not None:
        robot_cfg = G1_ROBOT_CFG.copy().replace(
            prim_path="/World/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),  # Centered position
                rot=(0.0, 0.0, 0.0, 1.0)  # No rotation initially
            )
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
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),  # Centered position
                rot=(0.0, 0.0, 0.0, 1.0)  # No rotation initially
            )
        )
        robot = Articulation(cfg=robot_cfg)
        scene_entities["robot"] = robot
    
    # Add camera
    camera = define_camera()
    scene_entities["camera"] = camera
    
    # Create objects in a semi-circle in front of the robot for better visibility
    radius = 2.5  # Distance from robot
    angle_step = np.pi / 6  # 30 degrees
    
    # Create objects at different positions around the robot
    object_configs = [
        # (object_name, object_type, color, size, height, angle_offset)
        ("Person", "cylinder", (0.8, 0.6, 0.6), (0.2, 1.8), 0.9, 0),           # Person directly in front
        ("Car", "cuboid", (0.1, 0.1, 0.8), (2.0, 1.0, 0.8), 0.4, angle_step),  # Car to the right
        ("Chair", "complex", (0.6, 0.4, 0.2), None, 0.5, -angle_step),          # Chair to the left
        ("RedCube", "cuboid", (1.0, 0.1, 0.1), (0.5, 0.5, 0.5), 1.0, angle_step*2),  # Red cube further right
        ("BlueCube", "cuboid", (0.1, 0.3, 1.0), (0.5, 0.5, 0.5), 1.0, -angle_step*2), # Blue cube further left
        ("TrafficLight", "complex", (0.2, 0.2, 0.2), None, 2.0, angle_step*3),  # Traffic light
    ]
    
    # Place objects around the robot
    for obj_name, obj_type, color, size, height, angle in object_configs:
        # Calculate position based on radius and angle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        if obj_type == "cylinder":
            radius_val, height_val = size
            cfg = sim_utils.CylinderCfg(
                radius=radius_val,
                height=height_val,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            cfg.func(f"/World/{obj_name}", cfg, translation=(x, y, height/2))
            print(f"[INFO]: Added a {obj_name} representation at position ({x:.1f}, {y:.1f}, {height/2:.1f})")
            
        elif obj_type == "cuboid":
            cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.1,
                    angular_damping=0.1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            )
            cfg.func(f"/World/{obj_name}", cfg, translation=(x, y, height/2))
            print(f"[INFO]: Added a {obj_name} representation at position ({x:.1f}, {y:.1f}, {height/2:.1f})")
            
        elif obj_type == "complex" and obj_name == "Chair":
            # Chair is made of two parts: seat and back
            seat_cfg = sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            seat_cfg.func(f"/World/{obj_name}/Seat", seat_cfg, translation=(x, y, height))
            
            backrest_cfg = sim_utils.CuboidCfg(
                size=(0.5, 0.1, 0.6),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            backrest_cfg.func(f"/World/{obj_name}/Back", backrest_cfg, translation=(x, y-0.25, height+0.3))
            print(f"[INFO]: Added a {obj_name} representation at position ({x:.1f}, {y:.1f}, {height:.1f})")
            
        elif obj_type == "complex" and obj_name == "TrafficLight":
            # Traffic light with box and three lights
            light_box_cfg = sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.6),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            light_box_cfg.func(f"/World/{obj_name}/Box", light_box_cfg, translation=(x, y, height))
            
            red_light_cfg = sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
            red_light_cfg.func(f"/World/{obj_name}/Red", red_light_cfg, translation=(x, y, height+0.2))
            
            yellow_light_cfg = sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            )
            yellow_light_cfg.func(f"/World/{obj_name}/Yellow", yellow_light_cfg, translation=(x, y, height))
            
            green_light_cfg = sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
            green_light_cfg.func(f"/World/{obj_name}/Green", green_light_cfg, translation=(x, y, height-0.2))
            print(f"[INFO]: Added a {obj_name} representation at position ({x:.1f}, {y:.1f}, {height:.1f})")
    
    # Add a few more COCO dataset objects that YOLOv8 can recognize
    
    # Add a bottle
    bottle_cfg = sim_utils.CylinderCfg(
        radius=0.1,
        height=0.3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2)),  # Green bottle
    )
    bottle_cfg.func("/World/Bottle", bottle_cfg, translation=(1.0, 1.5, 0.15))
    print(f"[INFO]: Added a bottle representation at position (1.0, 1.5, 0.15)")
    
    # Add a laptop (represented by a thin cuboid)
    laptop_cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.2, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),  # Gray laptop
    )
    laptop_cfg.func("/World/Laptop", laptop_cfg, translation=(1.2, -1.2, 0.51))
    print(f"[INFO]: Added a laptop representation at position (1.2, -1.2, 0.51)")
    
    return scene_entities


class YOLODetector:
    """Class to handle YOLOv8 object detection."""
    
    def __init__(self, model_name="yolov8s.pt", confidence=0.4, device=None):
        """Initialize the YOLOv8 detector."""
        self.available = YOLO_AVAILABLE
        if not self.available:
            return
        
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        try:
            # Download model if not available
            print(f"[INFO]: Loading YOLOv8 model '{model_name}' on device {self.device}...")
            self.model = YOLO(model_name)
            
            # Check if CUDA is being used
            if 'cuda' in self.device and torch.cuda.is_available():
                print(f"[INFO]: Using CUDA for YOLOv8 inference. GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("[INFO]: Using CPU for YOLOv8 inference.")
                
            print(f"[INFO]: YOLOv8 model '{model_name}' loaded successfully!")
            
            # Print model class names for reference
            print(f"[INFO]: Model can detect the following classes:")
            class_names = self.model.names if hasattr(self.model, 'names') else {}
            for idx, class_name in class_names.items():
                print(f"  - Class {idx}: {class_name}")
                
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
            # Convert to numpy array format for YOLOv8
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
            
            # Ensure image has correct channel format (H,W,C)
            if len(image_np.shape) == 2:
                # Grayscale to RGB
                image_np = np.stack([image_np, image_np, image_np], axis=2)
            elif len(image_np.shape) == 3 and image_np.shape[0] == 3:
                # (C,H,W) to (H,W,C)
                image_np = np.transpose(image_np, (1, 2, 0))
            
            # Run detection with confidence threshold on the specified device
            results = self.model(
                image_np, 
                conf=self.confidence,
                device=self.device,
                verbose=False  # Reduce console output noise
            )
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
            
            # Ensure image is properly formatted for YOLOv8
            if len(image_np.shape) == 2:
                # Grayscale to RGB
                image_np = np.stack([image_np, image_np, image_np], axis=2)
            elif len(image_np.shape) == 3 and image_np.shape[0] == 3:
                # (C,H,W) to (H,W,C)
                image_np = np.transpose(image_np, (1, 2, 0))
            
            # Customize the plot settings for clearer visibility
            image_with_boxes = result.plot(
                conf=True,         # Show confidence
                line_width=2,      # Thicker boxes
                font_size=12,      # Larger font
                labels=True,       # Show labels
                boxes=True,        # Show boxes
            )
            
            # Add a frame counter and object count overlay
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0, 128)  # Semi-transparent background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Count detected objects by class
            detected_classes = {}
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'cls'):
                for cls in result.boxes.cls:
                    class_name = self.class_names[int(cls)]
                    if class_name not in detected_classes:
                        detected_classes[class_name] = 0
                    detected_classes[class_name] += 1
            
            # Add object count text at the top of the image
            y_pos = 20
            text = f"Detected Objects: {sum(detected_classes.values())}"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw semi-transparent background
            overlay = image_with_boxes.copy()
            cv2.rectangle(overlay, (5, y_pos - 15), (5 + text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, image_with_boxes, 0.4, 0, image_with_boxes)
            
            # Draw text
            cv2.putText(image_with_boxes, text, (10, y_pos), font, font_scale, text_color, thickness)
            
            # List detected object classes and counts
            y_pos += 20
            for class_name, count in detected_classes.items():
                text = f"- {class_name}: {count}"
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Draw background for this line
                overlay = image_with_boxes.copy()
                cv2.rectangle(overlay, (5, y_pos - 15), (5 + text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, image_with_boxes, 0.4, 0, image_with_boxes)
                
                # Draw text
                cv2.putText(image_with_boxes, text, (10, y_pos), font, font_scale, text_color, thickness)
                y_pos += 20
            
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
        
        # Initialize with empty images - updated for higher resolution
        self.rgb_img = np.zeros((640, 640, 3))
        self.depth_img = np.zeros((640, 640))
        self.detection_img = np.zeros((640, 640, 3))
        
        # Store latest detection results
        self.detection_results = None
        
        # Initialize counter for saving images
        self.frame_count = 0
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        self.running = False
        
        # Create matplotlib figure for visualization - optimized for displaying larger images
        self.fig = Figure(figsize=(18, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        # Create subplots for RGB, depth images, and detections
        self.ax_rgb = self.fig.add_subplot(131)
        self.ax_depth = self.fig.add_subplot(132)
        self.ax_detection = self.fig.add_subplot(133)
        
        # Initialize image display with new dimensions and turn off axis to maximize image area
        self.rgb_plot = self.ax_rgb.imshow(np.zeros((640, 640, 3)))
        self.depth_plot = self.ax_depth.imshow(np.zeros((640, 640)), cmap='turbo')  # Better depth visualization
        self.detection_plot = self.ax_detection.imshow(np.zeros((640, 640, 3)))
        
        # Turn off axis ticks for a cleaner display
        self.ax_rgb.set_xticks([])
        self.ax_rgb.set_yticks([])
        self.ax_depth.set_xticks([])
        self.ax_depth.set_yticks([])
        self.ax_detection.set_xticks([])
        self.ax_detection.set_yticks([])
        
        # Set titles
        self.ax_rgb.set_title('RGB Camera View')
        self.ax_depth.set_title('Depth View')
        self.ax_detection.set_title('YOLOv8 Detections')
        
        # Adjust layout to minimize whitespace
        self.fig.tight_layout()
        
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
        plt.imsave(depth_filename, self.depth_img, cmap='turbo')
        
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
        # Set initial joint positions for better camera view
        robot_joints = {
            # Lower body joints - standing pose
            ".*_hip_pitch_joint": -0.10,  # Less bent at the hip
            ".*_knee_joint": 0.20,        # Less bent at the knee
            ".*_ankle_pitch_joint": -0.10, # Less ankle pitch
            
            # Upper body joints - arms in a relaxed position
            ".*_elbow_pitch_joint": 0.5,  # Less bent elbows
            
            # Head joints - facing straight ahead
            "neck_pitch_joint": 0.0,      # Head looking straight ahead
            "neck_yaw_joint": 0.0,        # Head centered left/right
            
            # Shoulder joints - arms down and slightly forward
            "left_shoulder_roll_joint": 0.1,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.1,
            "right_shoulder_pitch_joint": 0.2,
            
            # Hands relaxed
            "left_one_joint": 0.5,
            "right_one_joint": -0.5,
            "left_two_joint": 0.25,
            "right_two_joint": -0.25,
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
    
    # Improved robot movement pattern - slower, more deliberate rotation
    movement_period = 30.0  # seconds for a full movement cycle
    movement_angle = 0.0
    head_angle = 0.0
    head_period = 8.0  # faster head movement to scan the environment
    
    # Track FPS for YOLOv8 inference
    frame_count = 0
    detection_count = 0
    last_fps_time = time.time()
    detection_fps = 0
    
    try:
        print("\n[INFO]: Starting simulation with G1 humanoid robot camera and YOLOv8 detection...")
        while simulation_app.is_running() and sim_time < 60.0:  # Run for 60 seconds
            frame_start_time = time.time()
            
            # Process camera data if available
            try:
                # Access camera data via data.output attribute
                if hasattr(camera, 'data') and hasattr(camera.data, 'output') and camera.data.output is not None:
                    rgb_data = camera.data.output.get("rgb")
                    depth_data = camera.data.output.get("distance_to_image_plane")
                    
                    if rgb_data is not None and depth_data is not None:
                        frame_count += 1
                        
                        # Only run detection every other frame to maintain performance
                        if frame_count % 2 == 0 and yolo_detector is not None and yolo_detector.available:
                            detection_count += 1
                        
                        visualizer.update_images(rgb_data, depth_data)
                        
                        # Save data using replicator if enabled
                        if args_cli.save:
                            rep_writer.write(camera.data.output)
            except Exception as e:
                if step_count % 100 == 0:
                    print(f"Warning processing camera data: {e}")
            
            # Calculate and display FPS every second
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                detection_fps = detection_count / (current_time - last_fps_time)
                frame_fps = frame_count / (current_time - last_fps_time)
                if step_count % 30 == 0:
                    print(f"Camera FPS: {frame_fps:.1f}, Detection FPS: {detection_fps:.1f}")
                frame_count = 0
                detection_count = 0
                last_fps_time = current_time
            
            # Move the robot in a way that's more conducive to object detection
            if robot is not None:
                # Calculate new body rotation angle - slower movement
                movement_angle = (sim_time / movement_period) * 2 * np.pi
                
                # Calculate independent head movement for faster scanning
                head_angle = (sim_time / head_period) * 2 * np.pi
                
                # Get current position and orientation of the robot
                root_state = robot.data.root_pos_w.clone()[0]
                
                # Create rotation quaternion for body (around Z-axis)
                # We'll use a gentler side-to-side motion instead of full rotation
                angle_range = np.pi/3  # 60 degrees range of motion
                body_angle = np.sin(movement_angle) * angle_range  # Sinusoidal movement
                quat = torch.tensor([np.cos(body_angle/2), 0, 0, np.sin(body_angle/2)], 
                                   device=robot.data.root_pos_w.device)
                
                # Apply new position and rotation to the robot's root
                robot.write_root_pose_to_sim(torch.cat([root_state[:3].unsqueeze(0), quat.unsqueeze(0)], dim=1))
                
                # Move the robot's head independently to scan the environment
                if step_count % 10 == 0:  # Update head position less frequently
                    joint_pos = robot.get_joint_positions(joint_indices=None)
                    joint_vel = robot.get_joint_velocities(joint_indices=None)
                    
                    # Find neck yaw joint index
                    neck_yaw_idx = None
                    neck_pitch_idx = None
                    for i, name in enumerate(robot.data.joint_names):
                        if name == "neck_yaw_joint":
                            neck_yaw_idx = i
                        elif name == "neck_pitch_joint":
                            neck_pitch_idx = i
                    
                    # If we found the neck joints, animate them
                    if neck_yaw_idx is not None:
                        # Sinusoidal head movement - look left and right
                        yaw_angle = np.sin(head_angle) * 0.5  # Max 0.5 radians (about 30 degrees) each way
                        joint_pos[:, neck_yaw_idx] = yaw_angle
                    
                    if neck_pitch_idx is not None:
                        # Slight up and down movement
                        pitch_angle = np.sin(head_angle * 1.5) * 0.2  # Faster, smaller pitch movement
                        joint_pos[:, neck_pitch_idx] = pitch_angle
                    
                    # Apply the joint positions
                    robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
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
            
            # Print detection information periodically
            if step_count % 100 == 0 and YOLO_AVAILABLE and visualizer.detection_results is not None:
                # Get detection results
                result = visualizer.detection_results[0] if isinstance(visualizer.detection_results, list) else visualizer.detection_results
                # Print detected classes and confidence
                print(f"\n--- Step {step_count} (Time: {sim_time:.2f}s) ---")
                print("Detected objects:")
                detected_classes = {}
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'cls'):
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
        yolo_detector = YOLODetector(
            model_name=args_cli.model, 
            confidence=args_cli.confidence,
            device=args_cli.device
        )
    
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