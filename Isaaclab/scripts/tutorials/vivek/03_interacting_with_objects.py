#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tutorial 3: Interacting with Objects in IsaacLab
================================================

This tutorial demonstrates how to:
1. Spawn different primitive shapes in IsaacLab
2. Apply forces and torques to objects
3. Read object states (position, velocity)
4. Implement basic physics interactions

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/03_interacting_with_objects.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# Create an argument parser
parser = argparse.ArgumentParser(description="Tutorial on interacting with objects")
# Add standard AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Once the app is launched, we can import the rest of the modules and define our simulation."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene by creating ground plane, lights, and objects."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    
    # Create a parent for all objects
    prim_utils.create_prim("/World/Objects", "Xform")
    
    # Create a red cube
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Objects/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.5], 
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
    )
    cube_object = RigidObject(cfg=cube_cfg)
    
    # Create a green cylinder
    cylinder_cfg = RigidObjectCfg(
        prim_path="/World/Objects/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.1,
            height=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.5, 0.5], 
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
    )
    cylinder_object = RigidObject(cfg=cylinder_cfg)
    
    # Create a blue sphere
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/Objects/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.5, -0.5, 0.5], 
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
    )
    sphere_object = RigidObject(cfg=sphere_cfg)
    
    # Return scene entities
    scene_entities = {
        "cube": cube_object,
        "cylinder": cylinder_object,
        "sphere": sphere_object
    }
    return scene_entities


def run_simulator(sim: SimulationContext, entities: dict):
    """Runs the simulation loop with physics interactions."""
    # Extract scene entities for readability
    cube = entities["cube"]
    cylinder = entities["cylinder"]
    sphere = entities["sphere"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    step_count = 0
    max_steps = 1000
    
    # Simulate physics
    while simulation_app.is_running() and step_count < max_steps:
        # Apply forces and torques at specific intervals
        if step_count % 100 == 0:
            # Apply random force to cube
            force = np.random.uniform(-10.0, 10.0, size=3)
            # Convert numpy array to torch tensor
            force_tensor = torch.tensor([force], device=sim.device)
            # Apply force at the center of mass
            cube.set_external_force_and_torque(
                forces=force_tensor,
                torques=torch.zeros((1, 3), device=sim.device)
            )
            print(f"Applied force {force} to Cube")
            
            # Apply random torque to cylinder
            torque = np.random.uniform(-1.0, 1.0, size=3)
            torque_tensor = torch.tensor([torque], device=sim.device)
            cylinder.set_external_force_and_torque(
                forces=torch.zeros((1, 3), device=sim.device),
                torques=torque_tensor
            )
            print(f"Applied torque {torque} to Cylinder")
            
            # Apply both force and torque to sphere
            force = np.random.uniform(-5.0, 5.0, size=3)
            torque = np.random.uniform(-0.5, 0.5, size=3)
            force_tensor = torch.tensor([force], device=sim.device)
            torque_tensor = torch.tensor([torque], device=sim.device)
            sphere.set_external_force_and_torque(
                forces=force_tensor,
                torques=torque_tensor
            )
            print(f"Applied force {force} and torque {torque} to Sphere")
        
        # Get and print object positions and velocities every 50 steps
        if step_count % 50 == 0:
            # Positions in world frame
            cube_pos = cube.data.root_state_w[0, :3].cpu().numpy()
            cylinder_pos = cylinder.data.root_state_w[0, :3].cpu().numpy()
            sphere_pos = sphere.data.root_state_w[0, :3].cpu().numpy()
            
            # Linear velocities
            cube_vel = cube.data.root_state_w[0, 7:10].cpu().numpy()
            cylinder_vel = cylinder.data.root_state_w[0, 7:10].cpu().numpy()
            sphere_vel = sphere.data.root_state_w[0, 7:10].cpu().numpy()
            
            print(f"\nStep {step_count}:")
            print(f"Cube position: {cube_pos}, velocity: {cube_vel}")
            print(f"Cylinder position: {cylinder_pos}, velocity: {cylinder_vel}")
            print(f"Sphere position: {sphere_pos}, velocity: {sphere_vel}")
        
        # Write data to simulation
        for obj in entities.values():
            obj.write_data_to_sim()
        
        # Perform simulation step
        sim.step()
        
        # Update object states from simulation
        for obj in entities.values():
            obj.update(sim_dt)
        
        # Increment step counter
        step_count += 1
    
    print("\nSimulation complete!")


def main():
    """Main function."""
    print("Welcome to Isaac Lab! Let's interact with some objects.")
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    
    # Design scene by adding assets to it
    scene_entities = design_scene()
    
    # Reset the simulation to initialize physics
    sim.reset()
    
    # Reset each object to initialize them
    for obj in scene_entities.values():
        obj.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulation loop
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 