# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to work with deformable objects in Isaac Lab.

This tutorial helps beginners understand:
1. How to create and simulate cloth and soft bodies
2. How to interact with deformable objects
3. How to apply constraints to deformable objects

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/09_deformable_objects.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create an argument parser
parser = argparse.ArgumentParser(description="Tutorial on working with deformable objects")
# Add standard AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Once the app is launched, we can import the rest of the modules and define our simulation."""

import numpy as np
import os
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, RigidObjectCfg, DeformableObjectCfg
from isaaclab.scene.ground import GroundPlaneCfg
from isaaclab.scene.light import DomeLightCfg
from isaaclab.assets.deformable import DeformableObject


def main():
    """Main function that demonstrates deformable object simulation."""
    
    print("Welcome to Isaac Lab! Let's explore deformable objects.")
    
    # Step 1: Initialize the simulation context
    # Use a smaller time step for more stable deformable simulation
    sim_cfg = SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)
    
    # Step 2: Set up a camera view
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    
    # Step 3: Create a scene
    scene = InteractiveScene(sim)
    
    # Step 4: Add a ground plane
    ground_cfg = GroundPlaneCfg(
        size=np.array([10.0, 10.0]),
        color=np.array([0.2, 0.2, 0.2, 1.0])
    )
    scene.add_ground(ground_cfg)
    
    # Step 5: Add lighting for better visualization
    light_cfg = DomeLightCfg(
        intensity=1000.0,
        color=np.array([1.0, 1.0, 1.0])
    )
    scene.add_dome_light(light_cfg)
    
    # Step 6: Add rigid objects for the deformable objects to interact with
    # Create a table-like platform
    table_cfg = RigidObjectCfg(
        prim_path="/World/table",
        name="table",
        shape_cfg={"type": "box", "size": np.array([1.0, 1.0, 0.1])},
        init_state={"pos": np.array([0.0, 0.0, 0.5])},
        color=np.array([0.5, 0.3, 0.1, 1.0]),  # Brown
    )
    table = scene.add_rigid_object(table_cfg)
    
    # Add a small obstacle on the table
    obstacle_cfg = RigidObjectCfg(
        prim_path="/World/obstacle",
        name="obstacle",
        shape_cfg={"type": "box", "size": np.array([0.1, 0.1, 0.2])},
        init_state={"pos": np.array([0.0, 0.0, 0.8])},
        color=np.array([0.7, 0.7, 0.7, 1.0]),  # Gray
    )
    obstacle = scene.add_rigid_object(obstacle_cfg)
    
    # Step 7: Add a cloth (deformable object)
    # Configure the cloth parameters
    cloth_cfg = DeformableObjectCfg(
        prim_path="/World/cloth",
        name="cloth",
        # Using a pre-defined cloth model from Isaac Lab assets
        usd_path="{ISAACLAB_ASSETS}/deformable/cloth/cloth_square.usd",
        init_state={
            "pos": np.array([0.0, 0.0, 1.5]),  # Start above the table
            "scale": np.array([2.0, 2.0, 1.0])  # Scale to make it larger
        },
        # Physics properties for the cloth
        deformable_params={
            "solver_position_iteration_count": 8,  # Higher iteration for stable simulation
            "solver_velocity_iteration_count": 0,
            "enable_self_collision": True,         # Enable cloth self-collision
            "damping": 0.01                        # Add damping for stability
        }
    )
    
    # Add the cloth to the scene
    cloth = scene.add_deformable_object(cloth_cfg)
    
    # Step 8: Add a soft ball
    # Configure the soft ball
    soft_ball_cfg = DeformableObjectCfg(
        prim_path="/World/soft_ball",
        name="soft_ball",
        # Using a pre-defined soft ball model from Isaac Lab assets
        usd_path="{ISAACLAB_ASSETS}/deformable/soft_body/sphere.usd",
        init_state={
            "pos": np.array([0.5, 0.5, 1.5]),  # Position beside the cloth
            "scale": np.array([0.2, 0.2, 0.2])  # Scale to adjust size
        },
        # Physics properties for the soft body
        deformable_params={
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 0,
            "particle_mass_scale": 0.1,           # Lower mass for a lighter ball
            "damping": 0.05                       # Add damping for stability
        }
    )
    
    # Add the soft ball to the scene
    soft_ball = scene.add_deformable_object(soft_ball_cfg)
    
    # Step 9: Add a second cloth with fixed corners
    fixed_cloth_cfg = DeformableObjectCfg(
        prim_path="/World/fixed_cloth",
        name="fixed_cloth",
        usd_path="{ISAACLAB_ASSETS}/deformable/cloth/cloth_square.usd",
        init_state={
            "pos": np.array([0.0, -1.5, 1.0]),  # Different position
            "scale": np.array([1.0, 1.0, 1.0])
        },
        deformable_params={
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 0,
            "enable_self_collision": True,
            "damping": 0.01
        }
    )
    
    # Add the fixed cloth to the scene
    fixed_cloth = scene.add_deformable_object(fixed_cloth_cfg)
    
    # Reset the scene to initialize physics
    scene.reset()
    print("[INFO]: Scene setup complete!")
    
    # After reset, add constraints to fix the corners of the second cloth
    # This will create a hanging cloth effect
    
    # Get the particle indices for the corners
    num_particles = fixed_cloth.get_particle_count()
    # For a cloth, the particles are arranged in a grid
    grid_size = int(np.sqrt(num_particles))
    
    # Corner indices for a grid
    corner_indices = [
        0,                  # Top-left corner
        grid_size - 1,      # Top-right corner
        num_particles - grid_size,  # Bottom-left corner
        num_particles - 1   # Bottom-right corner
    ]
    
    # Fix only the top corners to create a hanging cloth
    fixed_cloth.set_fixed_particles([corner_indices[0], corner_indices[1]])
    print(f"Fixed the top corners of the cloth at indices {corner_indices[0]} and {corner_indices[1]}")
    
    # Step 10: Run the simulation
    step_count = 0
    while simulation_app.is_running() and step_count < 2000:  # Run for ~10 seconds
        # Print info every 200 steps
        if step_count % 200 == 0:
            print(f"\nStep {step_count}:")
            print(f"  Cloth particle count: {cloth.get_particle_count()}")
            print(f"  Soft ball particle count: {soft_ball.get_particle_count()}")
            
            # Get the center of mass position for the soft ball
            soft_ball_com = soft_ball.get_com_position()
            print(f"  Soft ball center of mass: {soft_ball_com}")
        
        # Apply an impulse to the soft ball at step 500
        if step_count == 500:
            print("\nApplying impulse to the soft ball!")
            particle_indices = list(range(soft_ball.get_particle_count()))
            impulse_vector = np.array([-1.0, 0.0, 0.0])  # Push left
            soft_ball.apply_impulse(particle_indices, impulse_vector)
        
        # Apply a force to a part of the cloth at step 1000
        if step_count == 1000:
            print("\nApplying force to the center of the cloth!")
            # Get the center particles of the cloth (approximate)
            cloth_particle_count = cloth.get_particle_count()
            center_particle = cloth_particle_count // 2
            # Apply a downward force
            force_vector = np.array([0.0, 0.0, -10.0])  # Push down
            cloth.apply_force([center_particle], force_vector)
        
        # Perform one simulation step
        scene.step()
        step_count += 1
    
    print("Simulation complete! You've worked with deformable objects in Isaac Lab.")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation application when done
    simulation_app.close() 