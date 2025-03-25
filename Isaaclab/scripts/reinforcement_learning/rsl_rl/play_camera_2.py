# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with added head camera visualization."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Add camera-related arguments
parser.add_argument("--with_camera", action="store_true", default=True, help="Enable head camera on G1 robot.")
parser.add_argument("--camera_fps", type=int, default=10, help="Camera update frequency in frames per second.")
parser.add_argument(
    "--save_images", action="store_true", default=False, 
    help="Save camera images to disk instead of just printing statistics."
)
parser.add_argument(
    "--save_dir", type=str, default="camera_images", 
    help="Directory to save camera images (relative to logs directory)."
)
parser.add_argument(
    "--debug", action="store_true", default=True,
    help="Print debugging information about environment and camera."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video or use head camera
if args_cli.video or args_cli.with_camera:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
from PIL import Image
from datetime import datetime

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def force_camera_update(env, dt):
    """Force camera update and provide extensive debug info."""
    if not hasattr(env.unwrapped, 'scene'):
        print("[DEBUG] Environment has no scene attribute")
        return False
    
    # List scene keys
    try:
        scene_keys = list(env.unwrapped.scene.keys())
        print(f"[DEBUG] Scene keys: {scene_keys}")
    except Exception as e:
        print(f"[DEBUG] Error listing scene keys: {e}")
        return False
    
    # Check for camera
    if 'head_camera' not in scene_keys:
        print("[DEBUG] No head_camera in scene keys")
        return False
    
    try:
        camera = env.unwrapped.scene['head_camera']
        print(f"[DEBUG] Camera type: {type(camera)}")
        print(f"[DEBUG] Camera dir: {dir(camera)}")
        
        # Try to force an update
        if hasattr(camera, 'update'):
            camera.update(dt)
            print("[DEBUG] Manually updated camera")
            
            # Check data after update
            if hasattr(camera, 'data'):
                print(f"[DEBUG] Camera data dir: {dir(camera.data)}")
                
                if hasattr(camera.data, 'output'):
                    print(f"[DEBUG] Camera output keys: {camera.data.output.keys()}")
                    
                    if 'rgb' in camera.data.output:
                        rgb_data = camera.data.output['rgb']
                        print(f"[DEBUG] RGB data shape: {rgb_data.shape}")
                        return True
                    else:
                        print("[DEBUG] No RGB key in output")
                else:
                    print("[DEBUG] Camera data has no output attribute")
            else:
                print("[DEBUG] Camera has no data attribute")
        else:
            print("[DEBUG] Camera has no update method")
            
    except Exception as e:
        print(f"[DEBUG] Error accessing camera: {e}")
    
    return False


def add_camera_to_env_cfg(env_cfg):
    """Add a camera to the environment configuration."""
    # Import inside function to avoid global import issues
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    
    # Debug the environment configuration structure
    print("\n=== ENVIRONMENT CONFIGURATION DEBUG ===")
    print(f"Scene class: {env_cfg.scene.__class__.__name__}")
    print(f"Scene attributes: {dir(env_cfg.scene)}")
    
    if not hasattr(env_cfg.scene, "robot"):
        print("[WARNING] Cannot add camera: No robot found in environment config")
        return
    
    print(f"Robot config: {env_cfg.scene.robot}")
    
    # Check for G1 specific link structure
    if hasattr(env_cfg.scene.robot, "prim_path"):
        print(f"Robot prim path: {env_cfg.scene.robot.prim_path}")
    
    # Check what link we should use - from the configs, we should use "torso_link"
    print(f"Adding camera to robot torso_link...")
    
    # Add head camera to the robot if it doesn't already exist
    if not hasattr(env_cfg.scene, "head_camera"):
        env_cfg.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
            update_period=1.0/args_cli.camera_fps,  # Convert FPS to time period
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.2, 0.0, 0.2),         # Forward and up from the torso
                rot=(0.0, 0.0, 0.0, 1.0),    # Default orientation
                convention="ros"              # Using ROS convention
            ),
        )
        print("[INFO] Added head camera to G1 robot torso_link")
    else:
        print("[INFO] head_camera already exists in scene config")


def debug_environment(env):
    """Debug the environment to understand its structure."""
    print("\n=== ENVIRONMENT DEBUG ===")
    print(f"Environment type: {type(env)}")
    print(f"Environment unwrapped type: {type(env.unwrapped)}")
    
    # Check if env has scene attribute
    if hasattr(env.unwrapped, 'scene'):
        print(f"Scene type: {type(env.unwrapped.scene)}")
        # Get scene keys safely without using 'in' operator
        try:
            scene_keys = list(env.unwrapped.scene.keys())
            print(f"Scene keys: {scene_keys}")
            
            # Check if head_camera is in scene
            if 'head_camera' in scene_keys:
                try:
                    camera = env.unwrapped.scene['head_camera']
                    print(f"Camera type: {type(camera)}")
                    print(f"Camera attributes: {dir(camera)}")
                    print(f"Camera data attributes: {dir(camera.data) if hasattr(camera, 'data') else 'No data attribute'}")
                    
                    if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                        print(f"Camera output keys: {camera.data.output.keys()}")
                    else:
                        print("Camera has no output data yet")
                except Exception as e:
                    print(f"Error accessing camera: {e}")
            else:
                print("No head_camera found in scene keys")
                
            # Debug robot structure if available
            if 'robot' in scene_keys:
                try:
                    robot = env.unwrapped.scene['robot']
                    print(f"Robot type: {type(robot)}")
                    if hasattr(robot, 'find_bodies'):
                        # Try to find torso_link to confirm it exists
                        torso_body_indices = robot.find_bodies("torso_link")
                        print(f"Torso link indices: {torso_body_indices}")
                        
                        # List all body names for reference
                        if hasattr(robot.data, 'body_names'):
                            print(f"Available body names: {robot.data.body_names}")
                except Exception as e:
                    print(f"Error debugging robot: {e}")
        except Exception as e:
            print(f"Error getting scene keys: {e}")
    else:
        print("Environment has no scene attribute")
        
    # Debug policy observation space if available
    if hasattr(env, 'observation_space'):
        print(f"Observation space: {env.observation_space}")


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Add camera to environment config if requested
    if args_cli.with_camera:
        add_camera_to_env_cfg(env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # Create image save directory if saving images
    image_save_dir = None
    if args_cli.save_images:
        image_save_dir = os.path.join(log_dir, args_cli.save_dir)
        os.makedirs(image_save_dir, exist_ok=True)
        print(f"[INFO] Camera images will be saved to: {image_save_dir}")

    # create isaac environment
    print(f"[INFO] Creating environment for task: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Debug environment if requested
    if args_cli.debug:
        debug_environment(env)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        if args_cli.debug:
            print("[INFO] Converted to single-agent environment")
            debug_environment(env)  # Debug again after conversion

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during execution.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    
    # Initialize camera debug vars
    last_camera_check = 0
    camera_found = False
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # Enhanced camera debugging for early steps
            if timestep < 100 and timestep % 5 == 0:  # Check frequently in the first 100 steps
                print(f"\n=== CAMERA DEBUG at step {timestep} ===")
                camera_available = force_camera_update(env, dt)
                if camera_available:
                    print("[DEBUG] CAMERA SUCCESSFULLY UPDATED WITH RGB DATA")
            
            # Regular camera checking at longer intervals
            elif args_cli.with_camera and timestep - last_camera_check > 20:
                last_camera_check = timestep
                camera_found = False
                
                # Try to access camera through scene
                if hasattr(env.unwrapped, 'scene'):
                    try:
                        scene_keys = list(env.unwrapped.scene.keys())
                        if 'head_camera' in scene_keys:
                            try:
                                camera = env.unwrapped.scene['head_camera']
                                if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                                    if 'rgb' in camera.data.output:
                                        camera_found = True
                                        rgb_data = camera.data.output['rgb'][0]
                                        
                                        # Convert to float for calculations
                                        rgb_data_float = rgb_data.float() / 255.0
                                        
                                        # Calculate simple statistics
                                        rgb_mean = torch.mean(rgb_data_float, dim=(0, 1)).cpu().numpy()
                                        
                                        # Print camera stats
                                        print(f"\n[Step {timestep}] Camera RGB mean: ({rgb_mean[0]:.3f}, {rgb_mean[1]:.3f}, {rgb_mean[2]:.3f})")
                                        
                                        # Save image if requested
                                        if args_cli.save_images and image_save_dir:
                                            # Convert tensor to numpy and then to PIL Image
                                            rgb_np = rgb_data.cpu().numpy().astype(np.uint8)
                                            img = Image.fromarray(rgb_np)
                                            
                                            # Save with timestamp and step number
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            img_path = os.path.join(image_save_dir, f"camera_{timestamp}_step{timestep:06d}.png")
                                            img.save(img_path)
                                            print(f"Saved camera image to: {img_path}")
                                    else:
                                        print(f"[Step {timestep}] Camera found but no RGB data available. Output keys: {camera.data.output.keys()}")
                                else:
                                    print(f"[Step {timestep}] Camera found but no data or output available")
                                    try:
                                        # Try to force an update
                                        camera.update(dt)
                                        print("Manually updated camera")
                                    except Exception as e:
                                        print(f"Error updating camera: {e}")
                            except Exception as e:
                                print(f"Error accessing camera: {e}")
                        else:
                            print(f"[Step {timestep}] No head_camera in scene keys: {scene_keys}")
                    except Exception as e:
                        print(f"Error accessing scene keys: {e}")
                else:
                    print(f"[Step {timestep}] Environment has no scene attribute")
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()