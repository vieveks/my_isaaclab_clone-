# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn objects in Isaac Lab.

This tutorial helps beginners understand:
1. How to spawn primitive shapes (cube, sphere)
2. How to spawn a ground plane
3. How to position objects in the scene

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/02_spawning_objects.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning objects in Isaac Lab")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def design_scene():
    """Designs the scene by spawning ground plane, light, and primitive shapes."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # spawn dome light for better visibility
    cfg_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.8, 0.8, 0.8),
    )
    cfg_light.func("/World/Light", cfg_light)
    
    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")
    
    # spawn a red cube with colliders and rigid body
    cfg_cube = sim_utils.CuboidCfg(
        size=(1.0, 1.0, 1.0),  # 1x1x1 cube
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red color
    )
    cfg_cube.func("/World/Objects/my_cube", cfg_cube, translation=(0.0, 0.0, 0.5))
    
    # spawn a blue sphere with colliders and rigid body
    cfg_sphere = sim_utils.SphereCfg(
        radius=0.5,  # Sphere with radius 0.5
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # Blue color
    )
    cfg_sphere.func("/World/Objects/my_sphere", cfg_sphere, translation=(2.0, 0.0, 0.5))


def main():
    """Main function."""
    
    print("Welcome to Isaac Lab! Let's spawn some objects.")
    
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera view from farther away to see all objects
    sim.set_camera_view([5.0, 5.0, 5.0], [0.0, 0.0, 0.0])
    
    # Design scene by adding assets to it
    design_scene()
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Simulation setup complete with a cube and a sphere!")
    
    # Simulate physics
    counter = 0
    while simulation_app.is_running() and counter < 1000:  # Run for ~10 seconds
        # perform simulation step
        sim.step()
        counter += 1
        
        # Print a message every 100 steps
        if counter % 100 == 0:
            print(f"Simulation running: {counter/100} seconds")
    
    print("Simulation complete! You've just spawned and simulated objects in Isaac Lab.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 