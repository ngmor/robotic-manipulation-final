#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
from traj_gen import TrajectoryGenerator
from common import generate_csv, calculate_Tse, calculate_Tsc, test_joint_limits, ZERO_TOL
from traj_gen import TrajectoryGenerator
from simulator import NextState
from control import FeedbackControl, get_velocities_from_twist

def simulate_youbot(Tse_des_ini, config_ini, Tsc_ini, Tsc_fin, dt, velocity_limits, kp, ki,
                    total_time=25, gripper_actuate_time=0.625, standoff_time=1.5, k=1):
    """
    Generates a CSV of simulated configurations of the youbot moving to pick up a cube.

    Also generated are the reference trajectory CSV and a CSV that tracks the error twist over 
    the length of the simulation.

    Args:
        Tse_des_ini (4x4 np array): SE(3) transform from space frame to end effector at its initial 
                                    desired location
        config_ini (np-array, 13-vector): initial configuration of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]
        Tsc_ini (4x4 np array): SE(3) transform from space frame to cube at it's initial location
        Tsc_fin (4x4 np array): SE(3) transform from space frame to cube at it's final location
        dt (float): timestep in seconds
        velocity_limits (np-array, 9-vector): velocity limits, in this order:
            [W1_lim, W2_lim, W3_lim, W4_lim, J1_lim, J2_lim, J3_lim, J4_lim, J5_lim]
        kp (float): proportional gain constant
        ki (float): integral gain constant
        total_time (int, optional): total time for youbot to complete its task. Defaults to 25.
        gripper_actuate_time (float, optional): time for the gripper to actuate in seconds.
                                                Defaults to 0.625.
        standoff_time (float, optional): time to move from the standoff position to the grasp 
                                         position in seconds. Defaults to 1.5.
        k (int, optional): number of trajectory reference configurations per dt seconds.
                           Defaults to 1.
    """    

    # grasp transform
    # Rotated around the y axis from the cube position by the specified angle
    # Translated in the positive z axis from the cube position by the specified height
    grasp_angle = 3*np.pi/4 # rad
    grasp_height = 0.0 # m
    Tce_grasp = np.array([
        [np.cos(grasp_angle),0,np.sin(grasp_angle),0],
        [0,1,0,0],
        [-np.sin(grasp_angle),0,np.cos(grasp_angle),grasp_height],
        [0,0,0,1]
    ])

    # grasp to standoff transform
    # Translated up from the grasp position by the specified standoff height
    standoff_height = 0.075 # m
    Tgrasp_standoff = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,standoff_height],
        [0,0,0,1]
    ])

    # calculate cube to standoff position transformation
    Tce_standoff = Tgrasp_standoff @ Tce_grasp

    # Joint position limits
    joint_limits = np.array([
        [-np.inf,np.inf], # J1
        [-1.3, 1.29], # J2
        [-np.pi/2, np.pi/2], # J3
        [-np.pi/2, np.pi/2], # J4
        [-np.pi/2, ZERO_TOL] # J5
    ]).T

    # Calculate reference trajectory
    print('Begin reference trajectory generation.')
    [ref_traj_tf, ref_traj_csv] = TrajectoryGenerator(Tse_des_ini, Tsc_ini, Tsc_fin, 
                                                      Tce_grasp, Tce_standoff, k, dt, total_time,
                                                      gripper_actuate_time, standoff_time)

    generate_csv('traj.csv',ref_traj_csv,folder='csv')
    print('Reference trajectory generation complete.')

    # Total number of iterations = total_time / (dt / k)
    # this is the size of the reference trajectory
    N = ref_traj_csv.shape[0]

    # We want to output a csv with total_time / dt lines, which is also N / k lines
    # Init configuration trajectory
    num_timesteps = int(N/k)
    config_traj_csv = np.zeros((num_timesteps, ref_traj_csv.shape[1]))
    Xerr_csv = np.zeros((num_timesteps, 6))
    
    # initial state
    current_config = np.copy(config_ini)

    # Init integral sum
    Xerr_integral_sum = np.array([0.]*6)

    j = 0

    print('Begin animation generation.')
    for i in range(N - 1):
        # Get current actual Tse
        Tse_act = calculate_Tse(current_config)

        # Get current desired Tse and gripper control from reference trajectory
        Tse_des = ref_traj_tf[i][0]
        gripper_control = ref_traj_tf[i][1]

        # Get next desired Tse from reference trajectory
        Tse_des_next = ref_traj_tf[i+1][0]

        valid_joints = False
        invalid_joint = np.array([False]*5)
        counter = 0

        while not valid_joints:
            # Calculate control law
            [twist_cmd, Xerr, Xerr_integral_sum] = FeedbackControl(Tse_act, Tse_des,Tse_des_next,
                                                                kp,ki,dt,k,Xerr_integral_sum)

            # Convert commanded twist to commanded velocities
            velocity_cmd = get_velocities_from_twist(twist_cmd, current_config, invalid_joint)

            # Simulate next state
            [next_config, next_velocities] = NextState(current_config, velocity_cmd, 
                                                        velocity_limits, dt, k)

            # Add gripper control
            next_config[-1] = gripper_control

            # Test if any joints violate joint limits
            joint_test = test_joint_limits(next_config, joint_limits)

            # If any joints are invalid, retry
            valid_joints = not np.any(joint_test)

            # Mark invalid joints for next loop
            invalid_joint = np.logical_or(invalid_joint, joint_test)

            counter += 1
            if counter > 100:
                print('Error, could not resolve without violating joint limit.')
                print(invalid_joint)
                print(i)
                exit()
        
        # store state every kth timestep
        if (i % k) == 0:
            config_traj_csv[j,:] = np.copy(current_config)
            Xerr_csv[j,:] = np.copy(Xerr)
            j += 1

        # update current config
        current_config = np.copy(next_config)

    print('Animation generation complete.')

    # Save to CSV
    print('Writing animation and errors to files.')
    generate_csv('simulate_youbot.csv',config_traj_csv,folder='csv')
    generate_csv('error.csv',Xerr_csv,folder='csv')
    print('Done.')



if __name__ == "__main__":

    # initial desired end effector transform
    Tse_des_ini = np.array([
        [0,0,1,0],
        [0,1,0,0],
        [-1,0,0,0.5],
        [0,0,0,1]
    ])

    # initial actual end effector transform
    # config_ini = np.array([0,-0.517,0,0,-0.7,0.7,-np.pi/2,0,0,0,0,0,0])  # matches initial desired
    config_ini = np.array([np.pi/6,-0.75,0,0,-0.7,0.7,-np.pi/2,0,0,0,0,0,0])

    # initial cube transform
    Tsc_ini = calculate_Tsc(1.,0.,0.)
    # Tsc_ini = calculate_Tsc(1.0,0.5,1.0)

    # final cube transform
    Tsc_fin = calculate_Tsc(0.,-1.,-np.pi/2.)
    # Tsc_fin = calculate_Tsc(0.25,-1.5,-1.0)

    # Joint velocity limits
    # [W1_lim, W2_lim, W3_lim, W4_lim, J1_lim, J2_lim, J3_lim, J4_lim, J5_lim]
    # velocity_limits = np.array([np.inf]*9)
    # velocity_limits = np.array([10,10,10,10,1,1,1,1,1])
    velocity_limits = np.array([20,20,20,20,2,2,2,2,2])

    # Gain constants
    kp = 3.0
    ki = 0.

    # define timing information
    dt = 0.01 # sec
    total_time = 15 # sec
    k = 2

    simulate_youbot(Tse_des_ini, config_ini, Tsc_ini, Tsc_fin, dt, velocity_limits, kp, ki,
                    total_time=total_time, k=k)