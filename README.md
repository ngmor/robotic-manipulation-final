# ME 449 Robotic Manipulation
This repository contains code for my solution to Northwestern University's ME449 Robotic Manipulation [capstone project](http://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2022).

## Introduction
This solution generates a CSV for animation/simulation in CoppeliaSim of a youBot mobile manipulator. The youBot’s task is to pick up a block at a specified starting position and place it at a specified ending position. First, a reference trajectory is generated. Then feedforward/feedback control is used to correct the youBot’s end effector position to follow that reference trajectory. Each timestep (of 0.01 sec) is simulated using first-order Euler integration. The output CSV of robot configurations at each timestep can be loaded into [CoppeliaSim Scene6_youBot_cube](http://hades.mech.northwestern.edu/index.php/CoppeliaSim_Introduction#Scene_6:_CSV_Mobile_Manipulation_youBot) for animation.

## Video

[Demo](https://user-images.githubusercontent.com/113186159/207950676-59f4c977-5b5b-4e97-9b56-85e5ac07f1c0.mp4)

## Software Architecture
This solution is packaged as several Python modules:
1. `simulator.py` – contains functions relating to [Milestone 1: youBot Kinematics Simulator](http://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2022#Milestone_1:_youBot_Kinematics_Simulator_and_csv_Output), which involves simulating robot states using first-order Euler integration.
2. `traj_gen.py` – contains functions relating to [Milestone 2: Reference Trajectory Generation](http://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2022#Milestone_2:_Reference_Trajectory_Generation), which involves generating a reference trajectory for the robot's end-effector to follow to complete the task.
3. `control.py` – contains functions relating to [Milestone 3: Feedforward/Feedback Control](http://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2022#Milestone_3:_Feedforward_Control), which involves employing feedforward/PI control to correct any deviations that occur from the reference trajectory.
4. `common.py` – contains shared and general use functions/constants.
5. `main.py` – contains the main function, `simulate_youbot`, that puts together the entire solution and example code for how to use it.

Each module contains example code of how to run its functions at the end, in the top-level code environment conditional statement (`if __name__ == “__main__”:`).

Also included is `plot_error.py`, which generates a plot of the error twist components over the course of the simulation. To generate these plots from the error CSVs created by the `simulate_youbot` function, edit the filename in line 6 of `plot_error.py` to point to the correct file path and run the script. The default filename should point to the most recently produced error file, if the the script that created that error file was run in the same directory as `plot_error.py`.

Python module dependencies:
- `numpy`
- `os`
- `matplotlib`
- `modern_robotics`

Outputs:
- `simulate_youbot.csv` – simulated trajectory of youBot for animation in CoppeliaSim
- `error.csv` – values of error twist components through the duration of the simulation
- `traj.csv` – the desired reference trajectory of the youBot’s end effector.

## Self-Collision Avoidance
Also built into the `simulate_youbot` function are joint limits intended to prevent the robot from colliding with itself. These joint limits were chosen by inspection, using the sliders in [CoppeliaSim Scene3_youbot](http://hades.mech.northwestern.edu/index.php/CoppeliaSim_Introduction#Scene_3:_Interactive_youBot) to view joint motion that might cause self collision.

These joint limits are not perfect, but for the end effector orientations requested by this task, are sufficient to prevent self-collision.

Importantly, singularity avoidance is not built into these joint limits. This was a conscious choice, as for some starting configurations of the robot, preventing the joints from reaching singularity (at near-zero joint angles) also prevented them from rotating to the necessary configurations to actually pick up the cube. The successful completion of the task and avoidance of self-collisions are higher priority than avoidance of singularities.

In order to implement joint limits, the project’s suggested method was used. First, feedforward/feedback control was calculated. Then the next timestep’s joint positions were simulated. If any joints exceed their joint limits, that joint’s column in the robot's Jacobian was set to 0 and feedforward/feedback control was recalculated. This worked very effectively in preventing self-collisions.

## Improvements
One area that could be improved in this solution would be trajectory timing. Right now, several time parameters (total time, gripper actuation time, standoff motion time) must be passed into the reference trajectory generation function in order for it to generate a reference trajectory. The behavior of the robot can be very sensitive to these timing parameters. Generally longer times aren’t an issue, but if the total time for the simulation is too short, the robot’s control algorithm will have to deal with very large errors due to high required speeds and will not work well.

One solution to this would be modifying the reference trajectory generation function to use the maximum speed of joints/wheels to determine a reasonable total time for the task and generate a reference trajectory accordingly. This would reduce the possibility of having the feedback/feedforward control malfunction due to large errors caused by high speed demands.
