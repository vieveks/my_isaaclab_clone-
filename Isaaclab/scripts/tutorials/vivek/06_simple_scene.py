# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple interactive scene with multiple objects.

In this tutorial, you'll learn how to:
1. Set up a scene with multiple objects
2. Configure different physics properties
3. Create dynamic interactions between objects
4. Add visual materials for better appearance

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/06_simple_scene.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Tutorial on creating a simple interactive scene")
# Append AppLauncher command-line arguments
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Configure a simple scene with various objects."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(10.0, 10.0),
            color=(0.3, 0.3, 0.3)
        )
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0)
        )
    )

    # A tower of stacked boxes
    box1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box1",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )

    box2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box2",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 0.2)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )

    box3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box3",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.8)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )

    # Add a sphere that will collide with the tower
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.6, 0.2),
                metallic=0.8,
                roughness=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )

    # Add some cylinders as obstacles
    cylinder1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder1",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.7)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.3, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )

    cylinder2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder2",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.7)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, -0.3, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )

    # Add a cone that will be dropped from above
    cone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.1),
                roughness=0.6
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.5, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        )
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator with a simple scene and dynamic interactions."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    
    # Create a tensor for the sphere's initial push
    sphere_velocity = torch.zeros((1, 6), device=sim.device)
    sphere_push_applied = False
    
    # Create variables for dynamic events
    tower_hit = False
    cone_dropped = False
    
    # Simulate physics
    while simulation_app.is_running() and sim_time < 20.0:  # Run for 20 seconds
        # Reset scene occasionally
        if step_count % 1000 == 0 and step_count > 0:
            # Reset object positions
            for obj_name in ["box1", "box2", "box3", "sphere", "cylinder1", "cylinder2", "cone"]:
                try:
                    scene[obj_name].reset()
                except KeyError:
                    print(f"Object {obj_name} not found in scene")
            
            # Reset simulation state
            sphere_push_applied = False
            tower_hit = False
            cone_dropped = False
            
            print(f"\n[INFO]: Resetting scene at {sim_time:.2f}s...")
        
        # Apply velocity to the sphere after 1 second to knock over the tower
        if sim_time > 1.0 and not sphere_push_applied:
            # Set linear velocity toward the tower
            sphere_velocity[0, 0] = -2.0  # x-direction velocity
            scene["sphere"].write_root_velocity_to_sim(sphere_velocity)
            sphere_push_applied = True
            print("\n[EVENT]: Sphere launched toward the tower!")
        
        # Check if the sphere has reached the tower
        if sphere_push_applied and not tower_hit:
            sphere_pos = scene["sphere"].get_world_pose()[0]
            if sphere_pos[0, 0] < 0.3:  # Sphere has reached near the tower
                tower_hit = True
                print("\n[EVENT]: Collision with tower!")
                
                # Add a slight upward force to Box3 for more dramatic effect
                box3_velocity = torch.zeros((1, 6), device=sim.device)
                box3_velocity[0, 2] = 1.0  # z-direction velocity
                scene["box3"].write_root_velocity_to_sim(box3_velocity)
        
        # Make the cone fall toward the center after a delay
        if sim_time > 3.0 and not cone_dropped:
            cone_velocity = torch.zeros((1, 6), device=sim.device)
            cone_velocity[0, 1] = -1.0  # y-direction velocity
            scene["cone"].write_root_velocity_to_sim(cone_velocity)
            cone_dropped = True
            print("\n[EVENT]: Cone is falling!")
        
        # Occasionally print information about object positions
        if step_count % 100 == 0:
            try:
                sphere_pos = scene["sphere"].get_world_pose()[0][0]
                box1_pos = scene["box1"].get_world_pose()[0][0]
                print(f"\n--- Step {step_count} (Time: {sim_time:.2f}s) ---")
                print(f"Sphere position: ({sphere_pos[0]:.2f}, {sphere_pos[1]:.2f}, {sphere_pos[2]:.2f})")
                print(f"Box1 position: ({box1_pos[0]:.2f}, {box1_pos[1]:.2f}, {box1_pos[2]:.2f})")
            except (KeyError, IndexError):
                print("\n[WARNING]: Could not read object positions")
        
        # Write data to simulation
        scene.write_data_to_sim()
        
        # Perform simulation step
        sim.step()
        
        # Update scene state
        scene.update(sim_dt)
        
        # Update simulation time and step count
        sim_time += sim_dt
        step_count += 1


def main():
    """Main function."""
    
    print("Welcome to Isaac Lab! Let's create a simple interactive scene.")
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set up a camera view appropriate for viewing the scene
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.0])
    
    # Create the scene with various objects
    scene_cfg = SimpleSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset the simulation
    sim.reset()
    
    print("[INFO]: Simple scene setup complete with multiple objects!")
    print("[INFO]: Running simulation with dynamic interactions...")
    
    # Run the simulator
    run_simulator(sim, scene)
    
    print("Simulation complete! You've created a simple interactive scene in Isaac Lab.")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation app
    simulation_app.close() 