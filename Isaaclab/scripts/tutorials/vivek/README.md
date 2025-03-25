# Vivek's Isaac Lab Tutorials

This directory contains a series of tutorials designed to help beginners learn how to use Isaac Lab step by step. Each tutorial builds on the previous one, introducing new concepts progressively.

## Tutorial List

1. **01_hello_isaaclab.py**: Introduction to the basic concepts of Isaac Lab
   - How to launch the Isaac Sim application
   - How to create a simulation context
   - How to run a simple simulation loop

2. **02_spawning_objects.py**: Learn how to add objects to your simulation
   - How to spawn primitive shapes (cube, sphere)
   - How to spawn a ground plane
   - How to position objects in the scene

3. **03_interacting_with_objects.py**: Explore object interactions
   - How to create and use rigid bodies
   - How to apply forces and torques to objects
   - How to get information about object state

4. **04_robotic_arm.py**: Working with robotic arms
   - How to spawn a Franka Emika Panda robot
   - How to control joint positions
   - How to read joint states and end-effector pose

5. **05_adding_sensors.py**: Adding sensors to your simulation
   - How to add and configure a camera
   - How to capture RGB and depth images
   - How to use ray casting for distance sensing

6. **06_simple_scene.py**: Using the scene management system
   - How to use the InteractiveScene class for better scene organization
   - How to add multiple objects, lighting, and robots to a scene
   - How to interact with scene objects efficiently

7. **07_basic_rl_env.py**: Creating a reinforcement learning environment
   - How to define a custom RL environment
   - How to set up observation and action spaces
   - How to implement rewards and episode termination logic
   - How to test an RL environment with random actions

8. **08_inverse_kinematics.py**: Advanced robot control with inverse kinematics
   - How to set up differential IK for a robot arm
   - How to control a robot in Cartesian space
   - How to make a robot follow a trajectory of waypoints
   - How to visualize targets in the scene

9. **09_deformable_objects.py**: Simulating deformable objects
   - How to create and configure cloth simulations
   - How to work with soft bodies
   - How to apply constraints to deformable objects
   - How to interact with deformable objects by applying forces

10. **10_pick_and_place.py**: Implementing a practical robotics task
    - How to create a complete pick-and-place application
    - How to use contact sensors for grasp detection
    - How to control a gripper and coordinate arm movements
    - How to implement a task as a state machine

11. **11_manager_based_envs.py**: Working with manager-based environments
    - How to create modular RL environments using managers
    - How to implement scene, observation, reward, reset, and termination managers
    - How to train environments with rsl_rl
    - Best practices and common pitfalls in RL training

12. **12_manager_based_locomotion.py**: Advanced manager-based environment focused on quadruped locomotion tasks, showing specialized observation design, reward functions, and physics configuration for legged robots.

13. **13_advanced_navigation.py**: Goal-directed navigation to specific coordinates using velocity-based control, demonstrating target visualization, adaptive movement, and path correction for legged robots.

14. **14_g1_navigation.py**: Adaptation of navigation concepts for the G1 wheeled robot, implementing differential drive control and visualizing robot paths, showing how the same navigation framework can be applied to different robot platforms.

15. **15_g1_inference.py**: Demonstration of a trained navigation policy (or simulated one) for a G1 robot navigating to specific objects in a predefined sequence, including target visualization, path tracking, and object interaction effects.

## How to Run These Tutorials

You can run each tutorial using the Isaac Lab launcher script:

### On Linux:
```bash
./isaaclab.sh -p scripts/tutorials/vivek/01_hello_isaaclab.py
```

### On Windows:
```cmd
isaaclab.bat -p scripts/tutorials/vivek/01_hello_isaaclab.py
```

## Key Concepts in Isaac Lab

Isaac Lab is organized around several key components:

1. **Simulation Context**: The main container for the simulation, handling physics, time stepping, and more.

2. **Spawners**: Tools to create objects in the simulation world (shapes, grounds, robots, etc.).

3. **Assets**: Handles to interact with objects in the simulation (rigid bodies, articulations, deformable objects).

4. **Scenes**: Higher-level abstractions for creating complete simulation environments.

5. **Environments**: Structures that integrate all components for reinforcement learning or other robotics tasks.

6. **Sensors**: Components that can perceive the simulation environment (cameras, ray casters, etc.).

7. **Controllers**: Components that implement control algorithms for robots (IK, OSC, etc.).

8. **Managers**: Modular components that separate different aspects of environment logic for better organization and reuse.

## Learning Path

This tutorial series follows a logical progression:

1. **Basics** (Tutorials 1-3): Learn how to set up a simulation and work with basic objects.

2. **Advanced Features** (Tutorials 4-5): Explore robots and sensors for more complex simulations.

3. **Integration** (Tutorials 6-7): Bring everything together with scenes and RL environments.

4. **Advanced Topics** (Tutorials 8-9): Dive into specialized areas like inverse kinematics and deformable objects.

5. **Practical Applications** (Tutorial 10): Apply all previous concepts to implement real-world robotics tasks.

6. **Advanced Framework Features** (Tutorials 11-14): Learn about IsaacLab's modular environment architecture and training.

7. **Deployment and Inference** (Tutorial 15): Use trained policies for practical applications.

## Next Steps

After completing these tutorials, you might want to explore:

- Implementing more advanced RL algorithms with your environment
- Working with more complex robots and articulated systems
- Creating multi-agent robotic simulations
- Using advanced sensors like LIDAR or tactile sensors
- Combining multiple techniques to create a complete robotic application

Check out the main Isaac Lab tutorials in the `scripts/tutorials/` directory for more advanced topics. 