# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script introduces the basic concepts of Isaac Lab.

This tutorial helps beginners understand:
1. How to launch the Isaac Sim application
2. How to create a simulation context
3. How to run a simple simulation loop

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/vivek/01_hello_isaaclab.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Introduction to Isaac Lab basics")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""
    
    print("Welcome to Isaac Lab! Let's get started with the basics.")
    
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera view
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Simulation setup complete!")
    
    # Simulate physics
    counter = 0
    while simulation_app.is_running():
        # perform simulation step
        sim.step()
        counter += 1
        
        # Print a message every 100 steps (approximately every 1 second)
        if counter % 100 == 0:
            print(f"Simulation has been running for ~{counter/100} seconds")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 