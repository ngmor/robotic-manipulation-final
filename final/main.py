#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
from traj_gen import TrajectoryGenerator
from common import generate_csv, calculate_Tse, calculate_Tsc
from traj_gen import TrajectoryGenerator
from simulator import NextState
from control import FeedbackControl, get_velocities_from_twist

def simulate_youbot(Tse_des_ini, config_ini, Tsc_ini, Tsc_fin, dt,
                    total_time=25, gripper_actuate_time=0.625, standoff_time=1.5, k=1):
    """TODO"""

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
    standoff_height = 0.075
    Tgrasp_standoff = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,standoff_height],
        [0,0,0,1]
    ])

    # calculate cube to standoff position transformation
    Tce_standoff = Tgrasp_standoff @ Tce_grasp

    # Joint velocity limits
    # TODO - make these inputs?
    # [W1_lim, W2_lim, W3_lim, W4_lim, J1_lim, J2_lim, J3_lim, J4_lim, J5_lim]
    # velocity_limits = np.array([1000000000]*9)
    # TODO - use real values
    velocity_limits = np.array([10,10,10,10,1,1,1,1,1])

    # Gain constants
    # TODO
    kp = 1.
    ki = 0.5

    # Calculate reference trajectory
    [ref_traj_tf, ref_traj_csv] = TrajectoryGenerator(Tse_des_ini, Tsc_ini, Tsc_fin, 
                                                      Tce_grasp, Tce_standoff, k, dt, total_time,
                                                      gripper_actuate_time, standoff_time)

    generate_csv('traj.csv',ref_traj_csv,folder='csv')

    # Init configuration trajectory
    config_traj_csv = np.zeros(ref_traj_csv.shape)
    
    # initial state
    current_config = np.copy(config_ini)

    # Init integral sum
    Xerr_integral_sum = np.array([0.]*6)

    N = config_traj_csv.shape[0]

    for i in range(N - 1):
        # store state
        config_traj_csv[i,:] = np.copy(current_config)

        # Get current actual Tse
        Tse_act = calculate_Tse(current_config)

        # Get current desired Tse and gripper control from reference trajectory
        Tse_des = ref_traj_tf[i][0]
        gripper_control = ref_traj_tf[i][1]

        # Get next desired Tse from reference trajectory
        Tse_des_next = ref_traj_tf[i+1][0]


        # Calculate control law
        [twist_cmd, Xerr_integral_sum] = FeedbackControl(Tse_act, Tse_des,Tse_des_next,
                                                         kp,ki,dt,Xerr_integral_sum)

        # Convert commanded twist to commanded velocities
        velocity_cmd = get_velocities_from_twist(twist_cmd, current_config)

        # Simulate next state
        [current_config, next_velocities] = NextState(current_config, velocity_cmd, 
                                                      velocity_limits, dt)

        # Add gripper control
        current_config[-1] = gripper_control
        

        # TODO - figure out how to use k in the other functions
        # TODO - store error
        # TODO - handle joint limits

    # Store final state
    config_traj_csv[i+1,:] = np.copy(current_config)

    # Save to CSV
    generate_csv('simulate_youbot.csv',config_traj_csv,folder='csv')



if __name__ == "__main__":

    # initial desired end effector transform
    Tse_des_ini = np.array([
        [0,0,1,0],
        [0,1,0,0],
        [-1,0,0,0.5],
        [0,0,0,1]
    ])

    # initial actual end effector transform
    # TODO - add error
    config_ini = np.array([np.pi/6,-0.75,0,0,-0.7,0.7,-np.pi/2,0,0,0,0,0,0])
    # config_ini = np.array([0,-0.517,0,0,-0.7,0.7,-np.pi/2,0,0,0,0,0,0])  # matches initial desired

    # initial cube transform
    Tsc_ini = calculate_Tsc(1.,0.,0.)
    # Tsc_ini = calculate_Tsc(1.5,0.5,np.pi/3)

    # final cube transform
    Tsc_fin = calculate_Tsc(0.,-1.,-np.pi/2.)
    # Tsc_fin = calculate_Tsc(0.25,-1.5,-np.pi/3.)


    # define timing information
    dt = 0.01 # sec
    total_time = 25 # sec

    simulate_youbot(Tse_des_ini, config_ini, Tsc_ini, Tsc_fin, dt, total_time=total_time)