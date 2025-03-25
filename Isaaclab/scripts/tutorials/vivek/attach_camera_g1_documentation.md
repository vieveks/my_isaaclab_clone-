# G1 Robot Head Camera Tutorial Documentation

## Overview

This document provides detailed explanations of the `attach_camera_g1.py` script, which demonstrates how to attach a camera to the G1 robot's head and process the camera's RGB image data in real-time.

## Purpose

The script serves as an educational example of:
- Attaching virtual sensors to robot models in IsaacLab
- Accessing and processing sensor data during simulation
- Basic computer vision concepts (color analysis)
- Simple robot animation through joint control

## Script Structure

The script is organized into several key sections:

1. **Setup and Configuration**
2. **Scene Definition**
3. **Simulation Loop**
4. **Camera Data Processing**
5. **Main Execution**

## Detailed Breakdown

### 1. Setup and Configuration (Lines 1-45)

```python
# Command-line argument parsing
parser = argparse.ArgumentParser(description="G1 robot with head camera visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
```

This section handles:
- Setting up command-line arguments
- Enforcing that the `--enable_cameras` flag is active (required for camera functionality)
- Initializing the Isaac simulation environment

**Key Point**: The script checks if cameras are enabled and automatically adds the flag if necessary:
```python
# Make sure cameras are enabled for this script
if not args_cli.enable_cameras:
    print("[WARNING] This script requires enabling cameras. Adding --enable_cameras flag automatically.")
    args_cli.enable_cameras = True
```

### 2. Scene Definition (Lines 54-91)

```python
@configclass
class G1CameraSceneCfg(InteractiveSceneCfg):
    # Configuration details...
```

This section:
- Creates a `G1CameraSceneCfg` class that defines all components of the simulation
- Sets up a ground plane, lighting, and robot configuration
- Configures the head camera with specific parameters

**Camera Configuration Details:**
- **Attachment Point**: The camera is attached to the robot's `torso_link`
- **Update Rate**: 10Hz (every 100ms)
- **Resolution**: 320×240 pixels
- **Position Offset**: (0.2, 0.0, 0.2) - forward and up from the torso
- **Orientation**: Default orientation, facing forward
- **Coordinate System**: Uses ROS convention

### 3. Simulation Loop (Lines 93-172)

```python
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Simulation code...
```

This function:
- Extracts references to the robot and camera from the scene
- Sets up joint position targets for robot animation
- Implements a continuous simulation loop

**Animation Logic**:
```python
# Create a simple animation by moving some joints
if count % 50 == 0:
    # Make the robot look around by moving its torso joint
    torso_joint_idx = robot.find_joints("torso_joint")[0]
    angle = 0.1 * torch.sin(torch.tensor(count / 50.0))
    joint_pos_targets[:, torso_joint_idx] = angle
```

This creates a simple "looking around" motion by oscillating the torso joint using a sine wave.

### 4. Camera Data Processing (Lines 145-172)

```python
# Get and display camera data every 10 steps
if count % 10 == 0 and "rgb" in head_camera.data.output:
    # Data processing code...
```

This section:
- Retrieves RGB data from the camera every 10 simulation steps
- Displays information about the image data (shape, type)
- Calculates image statistics (mean color values, min/max)
- Performs simple scene analysis based on color composition

**Scene Analysis Logic**:
```python
# Simple detection of what's in view (just based on average color)
if rgb_mean[2] > 0.6:  # Lots of blue - likely looking at sky
    print("Camera likely viewing the sky")
elif rgb_mean[1] > 0.5 and rgb_mean[0] < 0.3:  # Green dominant - grass or ground
    print("Camera likely viewing grass/ground")
```

This implements a basic scene classification based on color composition:
- High blue component suggests the camera is viewing the sky
- High green with low red suggests grass or ground
- Low color variance (all channels similar) suggests walls or uniform surfaces

### 5. Main Execution (Lines 175-198)

```python
def main():
    # Main function code...
```

This function:
- Sets up the simulation context
- Positions the main viewpoint camera for good visualization
- Initializes the scene with the configuration
- Runs the simulation loop and handles cleanup

## Usage

To run the script:

```bash
./isaaclab.sh -p scripts/tutorials/vivek/attach_camera_g1.py --enable_cameras
```

## Key Concepts

### 1. Camera Attachment

The camera is attached to the robot as a child of the torso link, with an offset to position it properly:

```python
head_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
    # ... additional configuration ...
    offset=CameraCfg.OffsetCfg(
        pos=(0.2, 0.0, 0.2),          # Forward and up from the torso
        rot=(0.0, 0.0, 0.0, 1.0),     # Default orientation (facing forward)
        convention="ros"               # Using ROS convention
    ),
)
```

### 2. Camera Data Access

Camera data is accessed through the scene's reference to the camera sensor:

```python
rgb_data = head_camera.data.output["rgb"][0]
```

The `[0]` index selects data from the first environment when multiple environments are simulated.

### 3. Simple Computer Vision

The script demonstrates basic image analysis by:
1. Converting the RGB data to floating point (normalized to [0,1])
2. Calculating statistics (mean, min, max)
3. Using simple thresholds to classify what the camera might be viewing

### 4. Robot Animation

The robot is animated using a simple sinusoidal pattern applied to a selected joint:

```python
angle = 0.1 * torch.sin(torch.tensor(count / 50.0))
joint_pos_targets[:, torso_joint_idx] = angle
```

## Extensions and Modifications

This script could be extended to:

1. **Save camera images** to disk for later analysis
2. **Add more advanced computer vision** like object detection
3. **Control the robot based on visual input** (visual servoing)
4. **Use depth information** by adding a depth camera
5. **Implement a more complex robot behavior** based on camera data

## Conclusion

The `attach_camera_g1.py` script provides a foundational example of integrating cameras with robots in IsaacLab. It demonstrates how to:
- Configure and attach cameras
- Access and process image data
- Perform simple analytics on camera data
- Create basic robot animations

This serves as a starting point for more advanced robot perception applications. 