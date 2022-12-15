#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
from common import CHASSIS_H0, generate_csv, ZERO_TOL

def NextState(positions, velocities, velocity_limits,dt,k,accelerations=None):
    """
    Determine the next state of the robot based on first order Euler integration.

    Args:
        positions (np-array, 13-vector): current position of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]
        velocities (np-array, 9-vector): control velocities, in this order:
            [W1d, W2d, W3d, W4d, J1d, J2d, J3d, J4d, J5d]
        max_velocities (np-array, 9-vector): velocity limits, in this order:
            [W1_lim, W2_lim, W3_lim, W4_lim, J1_lim, J2_lim, J3_lim, J4_lim, J5_lim]
        dt (float): timestep for integration
        accelerations (np-array, 9-vector, optional): accelerations. Defaults to None (0 for all).
            in this order:
            [W1dd, W2dd, W3dd, W4dd, J1dd, J2dd, J3dd, J4dd, J5dd]
    Returns:
        next_positions (np-array, 13-vector): next position of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]
        next_joint_and_wheel_vel (np-array, 9-vector): next velocities, in this order:
            [W1d, W2d, W3d, W4d, J1d, J2d, J3d, J4d, J5d]
    """    

    # Get actual timestep for this simulator:
    timestep = dt / k

    # Separate positions
    chassis_pos = np.array(positions[0:3])
    joint_pos = np.array(positions[3:8])
    wheel_pos = np.array(positions[8:12])
    gripper = np.array(positions[-1])
    joint_and_wheel_pos = np.hstack((wheel_pos,joint_pos))

    # Make sure velocity limits are magnitudes
    velocity_limits_mag = np.abs(velocity_limits)

    # Limit velocities with the input limits
    joint_and_wheel_vel = np.copy(velocities)
    joint_and_wheel_vel = np.where(joint_and_wheel_vel > velocity_limits_mag,
                                   velocity_limits_mag, joint_and_wheel_vel)
    joint_and_wheel_vel = np.where(joint_and_wheel_vel < -velocity_limits_mag,
                                   -velocity_limits_mag, joint_and_wheel_vel)

    # If accelerations weren't passed in, set them to 0
    if accelerations is None:
        joint_and_wheel_accel = np.array([0.]*len(joint_and_wheel_vel))
    else:
        joint_and_wheel_accel = np.copy(accelerations)

    # Use first order Euler integration to get new joint and wheel angles/speeds
    [next_joint_and_wheel_pos, next_joint_and_wheel_vel] = \
        mr.EulerStep(joint_and_wheel_pos, joint_and_wheel_vel, joint_and_wheel_accel, timestep)

    # Limit output velocities with the input limits
    next_joint_and_wheel_vel = np.where(next_joint_and_wheel_vel > velocity_limits_mag,
                                        velocity_limits_mag, next_joint_and_wheel_vel)
    next_joint_and_wheel_vel = np.where(next_joint_and_wheel_vel < -velocity_limits_mag,
                                        -velocity_limits_mag, next_joint_and_wheel_vel)

    # ODOMETRY
    # Get body twist and extract components
    wheel_vel = joint_and_wheel_vel[0:4]
    body_twist = np.linalg.pinv(CHASSIS_H0) @ wheel_vel
    omega_bz = body_twist[0]
    v_bx = body_twist[1]
    v_by = body_twist[2]

    # Get change in body frame state in unit time
    if np.isclose(omega_bz,0,atol=ZERO_TOL):
        delta_q_b = np.array([
            0,
            v_bx,
            v_by,
        ])
    else:
        delta_q_b = np.array([
            omega_bz,
            (v_bx*np.sin(omega_bz) + v_by*(np.cos(omega_bz) - 1)) / omega_bz,
            (v_by*np.sin(omega_bz) + v_bx*(1 - np.cos(omega_bz))) / omega_bz,
        ])

    # Correct for timestep
    delta_q_b *= timestep

    # Get change in fixed frame state
    phi = chassis_pos[0]
    delta_q_s = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)],
    ]) @ delta_q_b

    # calculate new chassis position
    next_chassis_pos = chassis_pos + delta_q_s

    next_wheel_pos = next_joint_and_wheel_pos[0:4]
    next_joint_pos = next_joint_and_wheel_pos[4:9]


    next_positions = np.hstack((next_chassis_pos, next_joint_pos, next_wheel_pos,gripper))

    return next_positions, next_joint_and_wheel_vel


if __name__ == "__main__":

    # Testing code

    positions = np.array([0]*13)
    velocities = np.array([0]*9)
    max_velocities = np.array([10]*9)
    dt = 0.01
    k = 1
    N = int(1.0/dt)

    position_list = np.zeros((N,13))
    next_positions = np.copy(positions)
    velocities[0:4] = [10,10,10,10]
    next_velocities = np.copy(velocities)
    for i in range(N):
        position_list[i,:] = next_positions
        [next_positions, next_velocities] = NextState(next_positions, next_velocities,
                                                      max_velocities,dt, k)
        

    generate_csv('simulate_x.csv',position_list,folder='csv')

    position_list = np.zeros((N,13))
    next_positions = np.copy(positions)
    velocities[0:4] = [-10,10,-10,10]
    next_velocities = np.copy(velocities)
    for i in range(N):
        position_list[i,:] = next_positions
        [next_positions, next_velocities] = NextState(next_positions, next_velocities,
                                                      max_velocities,dt, k)
        

    generate_csv('simulate_y.csv',position_list,folder='csv')

    position_list = np.zeros((N,13))
    next_positions = np.copy(positions)
    velocities[0:4] = [-10,10,10,-10]
    next_velocities = np.copy(velocities)
    for i in range(N):
        position_list[i,:] = next_positions
        [next_positions, next_velocities] = NextState(next_positions, next_velocities,
                                                      max_velocities,dt, k)

    generate_csv('simulate_rotate.csv',position_list,folder='csv')

    position_list = np.zeros((N,13))
    next_positions = np.copy(positions)
    velocities[0:4] = [-10,10,-10,10]
    max_velocities[0:4] = [5,5,5,5]
    next_velocities = np.copy(velocities)
    for i in range(N):
        position_list[i,:] = next_positions
        [next_positions, next_velocities] = NextState(next_positions, next_velocities,
                                                      max_velocities,dt, k)

    generate_csv('simulate_limited.csv',position_list,folder='csv')

    position_list = np.zeros((N,13))
    next_positions = np.copy(positions)
    velocities[0:9] = [0,0,0,0,1,1,1,1,1]
    next_velocities = np.copy(velocities)
    for i in range(N):
        position_list[i,:] = next_positions
        [next_positions, next_velocities] = NextState(next_positions, next_velocities,
                                                      max_velocities,dt, k)

    generate_csv('simulate_arm.csv',position_list,folder='csv')
