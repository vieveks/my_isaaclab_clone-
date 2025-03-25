# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create and train a manager-based environment with rsl_rl.

This tutorial helps beginners understand:
1. What manager-based environments are and their advantages
2. How to create a simple manager-based environment
3. How to set up training with rsl_rl
4. Best practices and common pitfalls in RL training

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/11_manager_based_envs.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create an argument parser
parser = argparse.ArgumentParser(description="Tutorial on manager-based environments and rsl_rl")
# Add standard AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Once the app is launched, we can import the rest of the modules and define our simulation."""

import os
import math
import numpy as np
import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.envs import (
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)
from omni.isaac.lab.managers import (
    SceneEntityManager,
    ObservationManager,
    RewardManager,
    ResetManager,
    TerminationManager,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.scene.ground import GroundPlaneCfg
from isaaclab.utils import configclass

# Try to import rsl_rl - we'll need it for training
try:
    from rsl_rl.algorithms.ppo import PPO
    from rsl_rl.modules.actor_critic import ActorCritic
    from rsl_rl.env.vec_env_rlgames import VecEnvRLGames
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    rsl_rl_available = True
except ImportError:
    print("WARNING: rsl_rl package not found. Training will be disabled.")
    print("To enable training, please install rsl_rl: pip install rsl_rl")
    rsl_rl_available = False


"""
INTRODUCTION TO MANAGER-BASED ENVIRONMENTS

Manager-based environments provide a modular approach to building reinforcement learning
environments. Unlike direct environments where all functionality is in a single class,
manager-based environments separate responsibilities into different manager components:

1. SceneEntityManager: Handles the scene objects and their configurations
2. ObservationManager: Defines what information is included in observations
3. RewardManager: Computes rewards based on environment state
4. ResetManager: Handles environment reset logic
5. TerminationManager: Determines when episodes should end

This modular approach has several advantages:
- Better code organization and maintainability
- Easier to reuse components across different environments
- Cleaner separation of concerns
- More flexible for complex environments
"""

# Step 1: Create a scene config for the cartpole
@configclass
class SimpleCartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple cartpole scene."""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2000.0),
    )
    
    # cartpole
    cartpole = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        usd_path="{ISAACLAB_ASSETS}/robots/cartpole/cartpole.usd",
    )

# Step 2: Create a simple scene entity manager
class SimpleCartpoleSceneManager(SceneEntityManager):
    """A scene manager for a simple cartpole environment.
    
    The cartpole consists of:
    - A cart that can move along the x-axis
    - A pole (cylinder) attached to the cart with a hinge joint
    
    The goal is to balance the pole in an upright position.
    """

    def __init__(self, env, device):
        """Initialize the scene manager."""
        super().__init__(env, device)

    def design_scene(self) -> InteractiveScene:
        """Create the scene with the cart, pole, and ground."""
        # Create a scene with the configuration
        scene_cfg = SimpleCartpoleSceneCfg(
            num_envs=self._env.num_envs,
            env_spacing=2.0
        )
        scene = InteractiveScene(scene_cfg, sim=self._env.sim)
        
        # Store references to scene objects for easy access in other managers
        self._scene_entities = {"cartpole": scene["cartpole"], "ground": scene["ground"]}
        
        return scene

    def get_params(self) -> dict:
        """Return parameters that other managers might need."""
        # Define parameters for the environment
        params = {
            "num_observations": 4,  # [cart_pos, cart_vel, pole_angle, pole_vel]
            "num_actions": 1,       # [cart_force]
            "max_force": 10.0,      # Maximum force that can be applied to the cart
            "reset_position_noise": 0.1,  # Noise to add when resetting (makes RL more robust)
            "reset_rotation_noise": 0.1,
            "reset_velocity_noise": 0.1,
        }
        return params


# Step 3: Create an observation manager
class SimpleCartpoleObservationManager(ObservationManager):
    """Manages the observations for the cartpole environment.
    
    Observations include:
    - Cart position
    - Cart velocity
    - Pole angle
    - Pole angular velocity
    """

    def __init__(self, env, device="cpu"):
        """Initialize the observation manager."""
        super().__init__(env, device)

    def _setup_terms(self) -> list:
        """Define the observation terms (what information to include in observations)."""
        # We'll include basic state information about the cartpole
        terms = [
            # Cart position (x-axis)
            ObsTerm(
                name="cart_position",
                func=self._compute_cart_position,
                dim=1,
            ),
            # Cart velocity
            ObsTerm(
                name="cart_velocity",
                func=self._compute_cart_velocity,
                dim=1,
            ),
            # Pole angle (in radians)
            ObsTerm(
                name="pole_angle",
                func=self._compute_pole_angle,
                dim=1,
            ),
            # Pole angular velocity
            ObsTerm(
                name="pole_velocity", 
                func=self._compute_pole_velocity, 
                dim=1
            ),
        ]
        return terms

    def _compute_cart_position(self):
        """Get the position of the cart along the x-axis."""
        # The cart position is the first DOF of the articulation
        cart_pos = self._scene_entities["cartpole"].get_joint_positions()[0]
        # Normalize the position by dividing by a reasonable maximum
        # This helps the neural network learn more efficiently
        normalized_pos = cart_pos / 5.0
        return normalized_pos.reshape(-1, 1)

    def _compute_cart_velocity(self):
        """Get the velocity of the cart along the x-axis."""
        # The cart velocity is the first DOF velocity of the articulation
        cart_vel = self._scene_entities["cartpole"].get_joint_velocities()[0]
        # Normalize the velocity by dividing by a reasonable maximum
        normalized_vel = cart_vel / 10.0
        return normalized_vel.reshape(-1, 1)

    def _compute_pole_angle(self):
        """Get the angle of the pole from vertical."""
        # The pole angle is the second DOF of the articulation
        pole_angle = self._scene_entities["cartpole"].get_joint_positions()[1]
        # Normalize the angle (it's already in radians)
        # We don't need to divide by 2π because the angle is already limited by the joint
        return pole_angle.reshape(-1, 1)

    def _compute_pole_velocity(self):
        """Get the angular velocity of the pole."""
        # The pole angular velocity is the second DOF velocity of the articulation
        pole_vel = self._scene_entities["cartpole"].get_joint_velocities()[1]
        # Normalize the angular velocity by dividing by a reasonable maximum
        normalized_ang_vel = pole_vel / 15.0
        return normalized_ang_vel.reshape(-1, 1)


# Step 4: Create reward config
@configclass
class RewardsCfg:
    """Reward terms for the cartpole MDP."""

    # Base reward for keeping the pole balanced
    base_reward = RewTerm(func="_base_reward", weight=1.0)
    
    # Penalty for pole angle deviation from vertical
    pole_angle_penalty = RewTerm(func="_pole_angle_penalty", weight=-1.0)
    
    # Penalty for cart moving too far from center
    cart_position_penalty = RewTerm(func="_cart_position_penalty", weight=-0.5)


# Step 5: Create termination config
@configclass
class TerminationsCfg:
    """Termination terms for the cartpole MDP."""

    # Episode timeout
    time_out = DoneTerm(func="_check_episode_length", time_out=True)
    
    # Pole fell over
    pole_angle_limit = DoneTerm(func="_check_pole_angle")
    
    # Cart out of bounds
    cart_out_of_bounds = DoneTerm(func="_check_cart_position")


# Step 6: Create event config
@configclass
class EventCfg:
    """Configuration for events in the cartpole environment."""

    # Reset the cart position
    reset_cart_position = EventTerm(
        func="_reset_cartpole",
        mode="reset",
    )


# Step 7: Create environment config
@configclass
class SimpleCartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the simple cartpole environment."""

    # Scene settings
    scene = SimpleCartpoleSceneCfg(num_envs=4, env_spacing=2.0)
    
    # MDP settings
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    # Post initialization
    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 10
        # Viewer settings
        self.viewer.eye = (4.0, 0.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # Simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation


# Step 8: Create a reward manager
class SimpleCartpoleRewardManager(RewardManager):
    """Manages the rewards for the cartpole environment.
    
    The reward structure is:
    - +1 for each step the pole remains upright
    - Penalties for large pole angles or cart positions
    """

    def __init__(self, env, device="cpu"):
        """Initialize the reward manager."""
        super().__init__(env, device)

    def _base_reward(self):
        """Give a constant reward for staying alive."""
        return torch.ones(self._num_envs, 1, device=self._device)

    def _pole_angle_penalty(self):
        """Penalty for pole deviation from vertical position."""
        # Get pole angle
        pole_angle = self._scene_entities["cartpole"].get_joint_positions()[:, 1]
        # Square the angle to penalize larger deviations more (both positive and negative)
        angle_penalty = pole_angle.pow(2)
        return angle_penalty.unsqueeze(1)

    def _cart_position_penalty(self):
        """Penalty for cart moving too far from center."""
        # Get cart position
        cart_pos = self._scene_entities["cartpole"].get_joint_positions()[:, 0]
        # Square the position to penalize larger deviations more
        position_penalty = cart_pos.pow(2)
        return position_penalty.unsqueeze(1)


# Step 9: Create a termination manager
class SimpleCartpoleTerminationManager(TerminationManager):
    """Manages termination conditions for the cartpole environment.
    
    The episode ends when:
    1. The pole falls over (angle > 0.25 radians)
    2. The cart moves too far from center (position > 2.0)
    3. Maximum episode length is reached
    """

    def __init__(self, env, device="cpu"):
        """Initialize the termination manager."""
        super().__init__(env, device)
        
        # Define specific thresholds for termination
        self.max_pole_angle = 0.25 * math.pi  # ~45 degrees
        self.max_cart_position = 2.0
        self.max_episode_length = int(self._env.cfg.episode_length_s / (self._env.cfg.sim.dt * self._env.cfg.decimation))

    def _check_pole_angle(self, env_ids=None):
        """Check if the pole has fallen over."""
        if env_ids is None:
            env_ids = slice(None)
            
        # Get pole angle
        pole_angle = self._scene_entities["cartpole"].get_joint_positions()[:, 1]
        # Check if angle exceeds threshold
        pole_fallen = torch.abs(pole_angle) > self.max_pole_angle
        
        return pole_fallen[env_ids].unsqueeze(1)

    def _check_cart_position(self, env_ids=None):
        """Check if the cart has moved too far from center."""
        if env_ids is None:
            env_ids = slice(None)
            
        # Get cart position
        cart_pos = self._scene_entities["cartpole"].get_joint_positions()[:, 0]
        # Check if position exceeds threshold
        cart_oor = torch.abs(cart_pos) > self.max_cart_position
        
        return cart_oor[env_ids].unsqueeze(1)

    def _check_episode_length(self, env_ids=None):
        """Check if the episode has reached maximum length."""
        if env_ids is None:
            env_ids = slice(None)
            
        # Create tensor for result
        terminated = torch.full(
            (self._num_envs, 1), 
            False, 
            device=self._device, 
            dtype=torch.bool
        )
        
        # Check which episodes have timed out
        timed_out = self._env.episode_length_buf >= self.max_episode_length
        terminated[env_ids] = timed_out[env_ids].unsqueeze(1)
        
        return terminated[env_ids]


# Step 10: Create a reset manager
class SimpleCartpoleResetManager(ResetManager):
    """Manages reset logic for the cartpole environment.
    
    This handles:
    - Initial reset of the environment
    - Resetting after episode termination
    - Adding randomness for better generalization
    """

    def __init__(self, env, device="cpu"):
        """Initialize the reset manager."""
        super().__init__(env, device)
        
        # Define reset noise parameters
        self.reset_position_noise = 0.1
        self.reset_rotation_noise = 0.1 * math.pi
        self.reset_velocity_noise = 0.1

    def _reset_cartpole(self, env_ids=None):
        """Reset the cartpole for the specified environments."""
        if env_ids is None:
            env_ids = slice(None)
            
        cartpole = self._scene_entities["cartpole"]
        
        if isinstance(env_ids, slice):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        
        # Reset cart position with small random noise
        cart_pos = torch.zeros(num_envs, device=self._device)
        cart_pos += torch.randn(num_envs, device=self._device) * self.reset_position_noise
        
        # Reset pole angle with small random noise around upright position
        pole_angle = torch.zeros(num_envs, device=self._device)
        pole_angle += torch.randn(num_envs, device=self._device) * self.reset_rotation_noise
        
        # Set the joint positions (cart position and pole angle)
        dof_pos = torch.zeros((num_envs, 2), device=self._device)
        dof_pos[:, 0] = cart_pos
        dof_pos[:, 1] = pole_angle
        
        # Set velocities with some noise
        dof_vel = torch.randn((num_envs, 2), device=self._device) * self.reset_velocity_noise
        
        # Apply the reset values to the joints
        cartpole.set_joint_positions(dof_pos, env_ids)
        cartpole.set_joint_velocities(dof_vel, env_ids)
        
        return None


# Step 11: Create the Cartpole environment
class SimpleCartpoleEnv(ManagerBasedRLEnv):
    """Complete cartpole environment using manager-based architecture."""

    def __init__(self, cfg, sim, headless=False):
        """Initialize the cartpole environment."""
        super().__init__(cfg, sim, headless)
        
    def _process_actions(self, actions):
        """Process actions before applying them to the environment."""
        # Scale the actions from [-1, 1] to actual forces
        scaled_actions = actions * 10.0
        
        # Apply the actions to the cart
        # We use a zero for the pole's joint since we don't control it directly
        cart_actions = torch.zeros((self._scene_manager.cartpole.count, 2),
                                  device=self._device)
        cart_actions[:, 0] = scaled_actions[:, 0]  # Only apply force to the cart
        
        # Set the actions for the articulation
        self._scene_manager.cartpole.apply_joint_efforts(cart_actions)


# Step 12: Create a training function
def train_cartpole():
    """Setup and start training the cartpole environment with rsl_rl."""
    if not rsl_rl_available:
        print("Skipping training because rsl_rl is not available")
        return
    
    print("Setting up training with rsl_rl...")
    
    # Create the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # Set up environment parameters
    num_envs = 4  # Number of parallel environments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the environment configuration
    env_cfg = SimpleCartpoleEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Create the environment
    env = ManagerBasedRLEnv(
        cfg=env_cfg,
        sim=sim,
        headless=True,  # Don't render during training
        scene_manager_class=SimpleCartpoleSceneManager,
        observation_manager_class=SimpleCartpoleObservationManager,
        reward_manager_class=SimpleCartpoleRewardManager,
        reset_manager_class=SimpleCartpoleResetManager,
        termination_manager_class=SimpleCartpoleTerminationManager,
    )
    
    # Initialize the environment
    env.reset()
    
    # Create the RL Games adapter for rsl_rl
    vec_env = VecEnvRLGames(env)
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Create the neural network configuration
    actor_critic_config = {
        "iterations_per_update": 10,  # Number of optimization iterations per update
        "clip_param": 0.2,            # PPO clipping parameter
        "discount_gamma": 0.99,       # Discount factor for future rewards
        "gae_lambda": 0.95,           # GAE-Lambda parameter for advantage estimation
        "entropy_coef": 0.01,         # Entropy coefficient in the loss
        "learning_rate": 3e-4,        # Learning rate
        "value_loss_coef": 1.0,       # Value loss coefficient
        "max_grad_norm": 1.0,         # Gradient clipping threshold
        "use_clipped_value_loss": True,  # Whether to use clipped value loss in PPO
        "schedule": "adaptive",       # Learning rate schedule
        
        # Network parameters
        "actor_hidden_dims": [64, 64],  # Actor network hidden layer sizes
        "critic_hidden_dims": [64, 64], # Critic network hidden layer sizes
        "activation": "elu",            # Activation function
    }
    
    # Create the actor-critic network
    actor_critic = ActorCritic(
        num_observations=env.observation_space.shape[0],
        num_actions=env.action_space.shape[0],
        actor_hidden_dims=actor_critic_config["actor_hidden_dims"],
        critic_hidden_dims=actor_critic_config["critic_hidden_dims"],
        activation=actor_critic_config["activation"],
    ).to(device)
    
    # Create the PPO algorithm
    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=actor_critic_config["clip_param"],
        num_learning_epochs=actor_critic_config["iterations_per_update"],
        gamma=actor_critic_config["discount_gamma"],
        lam=actor_critic_config["gae_lambda"],
        entropy_coef=actor_critic_config["entropy_coef"],
        value_loss_coef=actor_critic_config["value_loss_coef"],
        learning_rate=actor_critic_config["learning_rate"],
        max_grad_norm=actor_critic_config["max_grad_norm"],
        use_clipped_value_loss=actor_critic_config["use_clipped_value_loss"],
        schedule=actor_critic_config["schedule"],
    )
    
    # Create the runner
    runner = OnPolicyRunner(
        vec_env=vec_env,
        algorithm=ppo,
        num_transitions_per_env=env_cfg.episode_length_s * 100,  # Collect this many transitions per environment
        num_learning_epochs=10,  # Number of times to run through the collected data
        num_mini_batches=4,      # Number of mini-batches per epoch
        save_interval=10,        # Save model every 10 iterations
        experiment_name="cartpole_ppo",  # Name of the experiment for logging
        run_name="",             # Run name for Tensorboard
        max_iterations=100,      # Maximum number of training iterations
    )
    
    # Start training
    print("Starting training...")
    runner.learn(env_cfg.episode_length_s * 100 * 100)  # Train for this many total timesteps
    
    # Close the environment
    env.close()
    
    print("Training complete!")


"""
DOs and DON'Ts for RL Training with Manager-Based Environments
-------------------------------------------------------------

DOs:
1. DO normalize your observations to reasonable ranges (usually [-1, 1] or [0, 1])
2. DO use randomization in resets to make your policy more robust
3. DO tune your reward function carefully - it's the most important part of RL
4. DO monitor training progress with metrics and visualizations
5. DO implement early stopping for failed episodes to speed up training
6. DO use a reasonable number of parallel environments (4-16 is often good)
7. DO save and evaluate your model regularly

DON'Ts:
1. DON'T make rewards too sparse (agent won't learn) or too dense (may learn wrong behavior)
2. DON'T use very large neural networks for simple tasks - they'll be harder to train
3. DON'T forget to normalize actions if using continuous control
4. DON'T use a very small batch size - it makes training unstable
5. DON'T set the learning rate too high - start with 3e-4 and tune from there
6. DON'T forget to check for NaN values in observations and rewards
7. DON'T train for too few iterations - RL requires many samples
"""


def main():
    """Main function that demonstrates the basic usage of manager-based environments."""
    
    print("Welcome to tutorial on manager-based environments in Isaac Lab!")
    
    # Create the environment configuration
    env_cfg = SimpleCartpoleEnvCfg()
    env_cfg.scene.num_envs = 1  # Just one environment for demonstration
    
    # Create simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Create the environment
    env = ManagerBasedRLEnv(
        cfg=env_cfg,
        sim=sim,
        headless=False,  # Show visualization
        scene_manager_class=SimpleCartpoleSceneManager,
        observation_manager_class=SimpleCartpoleObservationManager,
        reward_manager_class=SimpleCartpoleRewardManager,
        reset_manager_class=SimpleCartpoleResetManager,
        termination_manager_class=SimpleCartpoleTerminationManager,
    )
    
    # Initialize the environment
    env.reset()
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Simulate with random actions
    print("Running simulation with random actions...")
    step_count = 0
    
    while simulation_app.is_running() and step_count < 1000:
        # Sample random actions
        actions = torch.rand((env.num_envs, env.action_space.shape[0]), device=env.device) * 2.0 - 1.0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Print information every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}:")
            print(f"  Observation: {obs['policy'][0].cpu().numpy()}")
            print(f"  Reward: {reward[0].item()}")
            
        step_count += 1
    
    # Close the environment
    env.close()
    
    # If rsl_rl is available, ask if the user wants to try training
    if rsl_rl_available:
        print("\nDo you want to train the cartpole with PPO? (y/n)")
        response = input().lower().strip()
        if response == 'y' or response == 'yes':
            train_cartpole()
    
    print("\nTutorial complete!")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation application when done
    simulation_app.close() 