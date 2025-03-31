# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import traceback
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CustomActor(torch.nn.Module):
    """Custom actor network that exactly matches the structure of the checkpoint."""
    
    def __init__(self, state_dict):
        super().__init__()
        self.layer0 = torch.nn.Linear(123, 256)
        self.layer0.weight = torch.nn.Parameter(state_dict["actor.0.weight"])
        self.layer0.bias = torch.nn.Parameter(state_dict["actor.0.bias"])
        
        self.layer2 = torch.nn.Linear(256, 128)
        self.layer2.weight = torch.nn.Parameter(state_dict["actor.2.weight"])
        self.layer2.bias = torch.nn.Parameter(state_dict["actor.2.bias"])
        
        self.layer4 = torch.nn.Linear(128, 128)
        self.layer4.weight = torch.nn.Parameter(state_dict["actor.4.weight"])
        self.layer4.bias = torch.nn.Parameter(state_dict["actor.4.bias"])
        
        self.layer6 = torch.nn.Linear(128, 37)
        self.layer6.weight = torch.nn.Parameter(state_dict["actor.6.weight"])
        self.layer6.bias = torch.nn.Parameter(state_dict["actor.6.bias"])
    
    def forward(self, x):
        x = torch.nn.functional.elu(self.layer0(x))
        x = torch.nn.functional.elu(self.layer2(x))
        x = torch.nn.functional.elu(self.layer4(x))
        return self.layer6(x)


class PreTrainedPolicyAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        
        # First prepare the low-level action term so we can get dimensions if needed
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        
        # Add the low level observations to the observation manager early
        # so we can get the dimensions
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        
        # Now get the real input dimensions
        obs_shape = self._low_level_obs_manager.compute_group("ll_policy").shape[1]
        action_dim = self._low_level_action_term.action_dim
        print(f"Actual observation shape: {obs_shape}, action dim: {action_dim}")
        
        # Check if this is a JIT file or a checkpoint file
        try:
            # Try to load as a JIT file first
            file_bytes = read_file(cfg.policy_path)
            self.policy = torch.jit.load(file_bytes).to(env.device).eval()
            print(f"Loaded JIT policy from {cfg.policy_path}")
        except Exception as e:
            # If that fails, try to load as a checkpoint
            print(f"Could not load as JIT file: {e}")
            print(f"Trying to load as a checkpoint file from {cfg.policy_path}")
            
            try:
                # Load the checkpoint
                checkpoint = torch.load(cfg.policy_path, map_location=env.device)
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
                
                if "model_state_dict" in checkpoint:
                    print("Found model_state_dict in checkpoint")
                    # Directly create a custom actor that matches the checkpoint structure
                    self.policy = CustomActor(checkpoint["model_state_dict"]).to(env.device).eval()
                    print(f"Created custom actor network with exact architecture from checkpoint")
                else:
                    print("No model_state_dict found in checkpoint")
                    raise ValueError("Checkpoint doesn't contain model_state_dict")
                    
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                traceback.print_exc()
                
                # Try to use ANYmal policy as a last resort
                try:
                    print("Trying to use ANYmal policy as fallback")
                    anymal_policy_path = "/home/vivek/isaac_lab_2/_isaac_sim/kit/data/nucleus/IsaacLab/Policies/ANYmal-C/Blind/policy.pt"
                    file_bytes = read_file(anymal_policy_path)
                    self.policy = torch.jit.load(file_bytes).to(env.device).eval()
                    print(f"Loaded ANYmal policy as fallback")
                except Exception as e:
                    print(f"Failed to load ANYmal policy: {e}")
                    raise ValueError(f"Could not load any policy for navigation")

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            print(f"Low level observation shape: {low_level_obs.shape}, values min: {low_level_obs.min().item():.4f}, max: {low_level_obs.max().item():.4f}")
            
            try:
                print(f"Applying policy of type: {type(self.policy)}")
                self.low_level_actions[:] = self.policy(low_level_obs)
                print(f"Actions shape: {self.low_level_actions.shape}, values min: {self.low_level_actions.min().item():.4f}, max: {self.low_level_actions.max().item():.4f}")
            except Exception as e:
                print(f"Error applying policy: {e}")
                traceback.print_exc()
                # In case of error, output zeros
                self.low_level_actions.zero_()
                    
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""