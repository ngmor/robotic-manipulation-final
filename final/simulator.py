#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
from common import chassis_H0

def NextState(positions, velocities, velocity_limits,dt,accelerations=None):
    """
    Determine the next state of the robot based on first order Euler integration.

    Args:
        current_config (np-array, 12-vector): current configuration of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4]
        velocities (np-array, 9-vector): control velocities, in this order:
            [J1d, J2d, J3d, J4d, J5d, W1d, W2d, W3d, W4d]
        max_velocities (np-array, 9-vector): velocity limits, in this order:
            [J1_lim, J2_lim, J3_lim, J4_lim, J5_lim, W1_lim, W2_lim, W3_lim, W4_lim]
        dt (float): timestep for integration
        accelerations (np-array, 9-vector, optional): accelerations. Defaults to None (0 for all).
            in this order:
            [J1dd, J2dd, J3dd, J4dd, J5dd, W1dd, W2dd, W3dd, W4dd]
    """    

    # Separate positions
    chassis_pos = np.array(positions[0:3])
    joint_and_wheel_pos = np.array(positions[3:12])

    # Make sure velocity limits are magnitudes
    velocity_limits_mag = np.abs(velocity_limits)

    # Limit velocities with the input limits
    joint_and_wheel_vel = np.copy(velocities)
    joint_and_wheel_vel = np.where(joint_and_wheel_vel > velocity_limits_mag, velocity_limits_mag, joint_and_wheel_vel)
    joint_and_wheel_vel = np.where(joint_and_wheel_vel < -velocity_limits_mag, -velocity_limits_mag, joint_and_wheel_vel)

    # If accelerations weren't passed in, set them to 0
    if accelerations is None:
        joint_and_wheel_accel = np.array([0.]*len(joint_and_wheel_vel))
    else:
        joint_and_wheel_accel = np.copy(accelerations)

    # Use first order Euler integration to get new joint and wheel angles/speeds
    [next_joint_and_wheel_pos, next_joint_and_wheel_vel] = mr.EulerStep(joint_and_wheel_pos, joint_and_wheel_vel, joint_and_wheel_accel, dt)

    # Limit output velocities with the input limits
    next_joint_and_wheel_vel = np.where(next_joint_and_wheel_vel > velocity_limits_mag, velocity_limits_mag, next_joint_and_wheel_vel)
    next_joint_and_wheel_vel = np.where(next_joint_and_wheel_vel < -velocity_limits_mag, -velocity_limits_mag, next_joint_and_wheel_vel)

    # TODO - Odometry

"""
A 12-vector representing the current configuration of the robot (3 variables for the chassis configuration, 5 variables for the arm configuration, and 4 variables for the wheel angles).
A 9-vector of controls indicating the wheel speeds u (4 variables) and the arm joint speeds \dot{\theta} (5 variables).
A timestep Δt.
A positive real value indicating the maximum angular speed of the arm joints and the wheels. For example, if this value is 12.3, the angular speed of the wheels and arm joints is limited to the range [-12.3 radians/s, 12.3 radians/s]. Any speed in the 9-vector of controls that is outside this range will be set to the nearest boundary of the range. If you don't want speed limits, just use a very large number. If you prefer, your function can accept separate speed limits for the wheels and arm joints.
"""

if __name__ == "__main__":

    positions = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
    velocities = np.array([10,-10,11,-11,9,-9,10,10,10])
    max_velocities = np.array([10,10,-10,10,10,10,10,10,10])
    dt = 0.01


    NextState(positions, velocities, max_velocities,dt)
    print(chassis_H0)