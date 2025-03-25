# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple reinforcement learning environment in Isaac Lab.

This tutorial helps beginners understand:
1. How to create a basic RL environment
2. How to define observation and action spaces
3. How to implement reward functions and reset logic
4. How to test the environment with random actions

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/07_basic_rl_env.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create an argument parser
parser = argparse.ArgumentParser(description="Tutorial on creating an RL environment")
# Add standard AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Once the app is launched, we can import the rest of the modules and define our simulation."""

import numpy as np
import gymnasium as gym
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.envs import DirectRLEnv, RLEnvCfg, VecEnvCfg
from isaaclab.scene import InteractiveScene, RigidObjectCfg
from isaaclab.scene.ground import GroundPlaneCfg


# Define our custom RL environment by extending the DirectRLEnv class
class CubeBalancingEnv(DirectRLEnv):
    """A simple RL environment where the goal is to balance a cube on a platform."""

    def __init__(self, cfg=None, sim=None, headless=False):
        """Initialize the environment."""
        
        # Call the parent class constructor
        super().__init__(cfg, sim, headless)
        
        # Set up default parameters
        self.max_episode_length = 500  # 5 seconds
        self.cube_fall_threshold = 0.1  # Height below which we consider the cube fallen
        
        # Track the episode step count
        self.current_step = 0
        
        # Track if the cube has fallen off the platform
        self.cube_fallen = False
    
    def _design_scene(self) -> InteractiveScene:
        """Create the scene with a platform and a cube."""
        
        # Create an interactive scene
        scene = InteractiveScene(self._sim)
        
        # Add a ground plane
        ground_cfg = GroundPlaneCfg(
            size=np.array([10.0, 10.0]),
            color=np.array([0.2, 0.2, 0.2, 1.0])
        )
        scene.add_ground(ground_cfg)
        
        # Add a platform (a thin, wide cube)
        platform_cfg = RigidObjectCfg(
            prim_path="/World/platform",
            name="platform",
            shape_cfg={"type": "box", "size": np.array([1.0, 1.0, 0.1])},
            init_state={"pos": np.array([0.0, 0.0, 0.05])},
            color=np.array([0.8, 0.8, 0.8, 1.0])  # Light gray
        )
        self.platform = scene.add_rigid_object(platform_cfg)
        
        # Add a cube that will be balanced on the platform
        cube_cfg = RigidObjectCfg(
            prim_path="/World/balancing_cube",
            name="balancing_cube",
            shape_cfg={"type": "box", "size": np.array([0.2, 0.2, 0.2])},
            init_state={"pos": np.array([0.0, 0.0, 0.25])},  # Start above platform
            color=np.array([1.0, 0.0, 0.0, 1.0])  # Red
        )
        self.cube = scene.add_rigid_object(cube_cfg)
        
        return scene
    
    def _allocate_buffers(self):
        """Allocate observation and action buffers."""
        
        # Define observation space: [platform_position(3), platform_orientation(3), 
        #                            cube_position(3), cube_orientation(3), cube_velocity(3)]
        obs_dim = 15
        self._obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        
        # Define action space: [platform_velocity_x, platform_velocity_y]
        # We'll move the platform in the XY plane to balance the cube
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        
        # Allocate buffers
        self._actions = np.zeros((self.num_envs, self._action_space.shape[0]), dtype=np.float32)
        self._observations = np.zeros((self.num_envs, self._obs_space.shape[0]), dtype=np.float32)
        self._rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._dones = np.zeros(self.num_envs, dtype=np.bool_)
        self._truncated = np.zeros(self.num_envs, dtype=np.bool_)
        self._extras = [{} for _ in range(self.num_envs)]
    
    def _compute_observations(self):
        """Compute the current observations."""
        
        # Get platform state
        platform_pos, platform_quat = self.platform.get_world_pose()
        platform_rot = np.array(self._quat_to_euler(platform_quat))
        
        # Get cube state
        cube_pos, cube_quat = self.cube.get_world_pose()
        cube_rot = np.array(self._quat_to_euler(cube_quat))
        cube_vel = self.cube.get_linear_velocity()
        
        # Combine all data into the observation
        obs = np.concatenate([platform_pos, platform_rot, cube_pos, cube_rot, cube_vel])
        self._observations[0] = obs
        
        return self._observations
    
    def _quat_to_euler(self, quat):
        """Convert quaternion to Euler angles."""
        # Very simplified, in a real application you'd use a proper conversion
        # For this example, we'll just use the last 3 components
        return quat[1:4]
    
    def _compute_rewards(self):
        """Compute the reward for the current state."""
        
        # Get cube position
        cube_pos, _ = self.cube.get_world_pose()
        
        # Get platform position
        platform_pos, _ = self.platform.get_world_pose()
        
        # Calculate distance from cube to center of platform in XY plane
        distance_xy = np.sqrt((cube_pos[0] - platform_pos[0])**2 + 
                             (cube_pos[1] - platform_pos[1])**2)
        
        # Check if cube has fallen off the platform
        self.cube_fallen = cube_pos[2] < (platform_pos[2] + self.cube_fall_threshold)
        
        # Reward strategy:
        # 1. Negative reward based on XY distance from platform center (encourage centering)
        # 2. Large negative reward if cube falls off platform
        # 3. Small positive reward for each step the cube stays on the platform
        
        if self.cube_fallen:
            # Cube has fallen - large negative reward
            reward = -10.0
        else:
            # Cube is still on platform
            # Negative reward for distance from center, plus small positive reward for staying on
            reward = -distance_xy + 0.1
        
        self._rewards[0] = reward
        return self._rewards
    
    def _compute_dones(self):
        """Determine if the episode is done."""
        
        # Episode is done if:
        # 1. Cube has fallen off the platform
        # 2. Maximum episode length is reached
        
        done = self.cube_fallen or (self.current_step >= self.max_episode_length)
        self._dones[0] = done
        
        # Truncated if we hit max episode length but not failed
        self._truncated[0] = (self.current_step >= self.max_episode_length) and not self.cube_fallen
        
        return self._dones, self._truncated
    
    def _apply_actions(self, actions):
        """Apply the actions to the platform."""
        
        # Scale actions to the desired velocity range
        max_vel = 1.0  # maximum velocity in m/s
        platform_vel_x = actions[0, 0] * max_vel
        platform_vel_y = actions[0, 1] * max_vel
        
        # Set platform velocity
        self.platform.set_linear_velocity([platform_vel_x, platform_vel_y, 0.0])
    
    def reset(self, env_ids=None, options=None):
        """Reset the environment."""
        
        # Reset the scene
        self._scene.reset()
        
        # Reset episode variables
        self.current_step = 0
        self.cube_fallen = False
        
        # Reset platform position and velocity
        self.platform.set_world_pose(
            position=[0.0, 0.0, 0.05],
            orientation=[1.0, 0.0, 0.0, 0.0]
        )
        self.platform.set_linear_velocity([0.0, 0.0, 0.0])
        
        # Reset cube position
        # Add a small random offset to make it more challenging
        x_offset = np.random.uniform(-0.1, 0.1)
        y_offset = np.random.uniform(-0.1, 0.1)
        self.cube.set_world_pose(
            position=[x_offset, y_offset, 0.25],
            orientation=[1.0, 0.0, 0.0, 0.0]
        )
        self.cube.set_linear_velocity([0.0, 0.0, 0.0])
        
        # Compute initial observations
        observations = self._compute_observations()
        
        # Reset extras
        self._extras = [{} for _ in range(self.num_envs)]
        
        return observations, self._extras
    
    def step(self, actions):
        """Take a step in the environment."""
        
        # Apply actions
        self._actions = actions.copy()
        self._apply_actions(self._actions)
        
        # Step the simulation
        self._scene.step()
        
        # Increment step counter
        self.current_step += 1
        
        # Compute observations
        observations = self._compute_observations()
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Compute dones
        dones, truncated = self._compute_dones()
        
        return observations, rewards, dones, truncated, self._extras


def main():
    """Main function to test the RL environment."""
    
    print("Welcome to Isaac Lab! Let's create a basic RL environment.")
    
    # Step 1: Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # Step 2: Set up a camera view
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])
    
    # Step 3: Configure the RL environment
    env_cfg = RLEnvCfg(
        seed=42,  # Random seed for reproducibility
        physics_dt=0.01,  # Physics time step
        rendering_dt=0.01,  # Rendering time step
    )
    
    vec_env_cfg = VecEnvCfg(
        num_envs=1,  # Just one environment for now
    )
    
    # Step 4: Create and initialize the RL environment
    env = CubeBalancingEnv(
        cfg={"env": env_cfg, "vec_env": vec_env_cfg},
        sim=sim,
        headless=False  # We want to see the simulation
    )
    
    # Step 5: Test the environment with random actions
    observations, info = env.reset()
    print("\nInitial observation shape:", observations.shape)
    
    total_reward = 0.0
    num_steps = 0
    
    # Run for 300 steps (3 seconds) or until done
    while num_steps < 300 and simulation_app.is_running():
        # Generate random actions
        actions = np.random.uniform(-1.0, 1.0, size=(1, 2))
        
        # Take a step in the environment
        observations, rewards, dones, truncated, info = env.step(actions)
        
        # Accumulate reward
        total_reward += rewards[0]
        num_steps += 1
        
        # Print progress every 50 steps
        if num_steps % 50 == 0:
            print(f"Step {num_steps}, Current Reward: {rewards[0]:.2f}, Total Reward: {total_reward:.2f}")
        
        # Check if episode is done
        if dones[0]:
            print(f"\nEpisode ended after {num_steps} steps")
            print(f"Final Total Reward: {total_reward:.2f}")
            print(f"Reason: {'Cube fell' if not truncated[0] else 'Maximum steps reached'}")
            break
    
    print("\nSimulation complete! You've created a basic RL environment in Isaac Lab.")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation application when done
    simulation_app.close() 