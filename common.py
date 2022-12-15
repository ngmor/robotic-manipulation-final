#!/usr/bin/env python3

import numpy as np
import modern_robotics as mr
import os

ZERO_TOL = 1e-4

def generate_csv(filename,matrix,folder=''):
    """
    Generate csv from input matrix.

    Args:
        filename (str): name of file to output
        matrix (nxn np.array): Matrix to output to CSV
        folder (str, optional): Folder to place the file into. Defaults to ''.
    """    
    if folder != '':
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder += '/'

    filepath = folder + filename
    np.savetxt(filepath,matrix,delimiter=',',fmt='%10.6f')

def generate_hi(r,x,y,beta,gamma,phi):
    """
    Generate h_i for a wheel for an omniwheel robot.

    Args:
        r (float): wheel radius
        x (float): x coordinate of wheel in robot chassis frame
        y (float): y coordinate of wheel in robot chassis frame
        beta (float): angle between the positive x-axis of the robot chassis frame and the movement
                      direction caused by positive rotation of the wheel
        gamma (float): angle between the sliding direction of the wheel and a direction
                       perpendicular to the movement direction caused by positive rotation of the
                       wheel
        phi (float): chassis orientation

    Returns:
        np-array 3-vector: h_i for the wheel
    """    

    return (1 / (r*np.cos(gamma))) * np.array([
        x*np.sin(beta + gamma) - y*np.cos(beta + gamma),
        np.cos(beta + gamma + phi),
        np.sin(beta + gamma + phi),
    ]).T

def generate_H0(wheels):
    """
    Generate the H(0) matrix for an omnidirectional robot, given descriptions of the wheels.

    Args:
        wheels (nx4 2D array): a 2D array of wheel descriptions: [r,x,y,beta,gamma]

    Returns:
        (nx3 np-array): H0 for the omnidirectional robot.
    """    
    H0 = np.zeros((len(wheels),3))

    for i,wheel in enumerate(wheels):
        r = wheel[0]
        x = wheel[1]
        y = wheel[2]
        beta = wheel[3]
        gamma = wheel[4]
        
        hi = generate_hi(r,x,y,beta,gamma,0)

        H0[i,:] = hi

    return H0

def _generate_chassis_H0():
    """Generate the H0 matrix for the youbot."""

    r = 0.0475
    l = 0.47/2
    w = 0.3/2
    H0 = generate_H0([
        [r,l,w,0,-np.pi/4],
        [r,l,-w,0,np.pi/4],
        [r,-l,-w,0,-np.pi/4],
        [r,-l,w,0,np.pi/4],
    ])

    return H0

# H0 matrix for the chassis
CHASSIS_H0 = _generate_chassis_H0()

# constant transformation from b frame to 0 frame
Tb0 = np.array([
    [1,0,0,0.1662],
    [0,1,0,0],
    [0,0,1,0.0026],
    [0,0,0,1],
])

# Transformation between 0 frame and end effector frame at home configuration
M0e = np.array([
    [1,0,0,0.033],
    [0,1,0,0],
    [0,0,1,0.6546],
    [0,0,0,1],
])

# Arm Jacobian at home configuration
Jarm_home = np.array([
    [0, 0, 1, 0, 0.033, 0],
    [0, -1, 0, -0.5076, 0, 0],
    [0, -1, 0, -0.3526,0,0],
    [0, -1, 0, -0.2176,0,0],
    [0, 0, 1, 0,0,0],
]).T

def calculate_Tsb(config):
    """
    Calculate the SE(3) transformation from the space frame to the chassis frame given the current
    robot config.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]

    Returns:
        4x4 np-array: SE(3) transformation from the space frame to the chassis frame, Tsb
    """

    phi = config[0]
    x = config[1]
    y = config[2]

    return np.array([
        [np.cos(phi), -np.sin(phi), 0, x],
        [np.sin(phi), np.cos(phi), 0, y],
        [0,0,1,0.0963],
        [0,0,0,1],
    ])

def calculate_Ts0(config):
    """
    Calculate the SE(3) transformation from the space frame to the arm base link frame given the
    current robot config.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]

    Returns:
        4x4 np-array: SE(3) transformation from the space frame to the arm base link frame, Ts0
    """

    Tsb = calculate_Tsb(config)
    return Tsb @ Tb0

def calculate_T0e(config):
    """
    Calculate the SE(3) transformation from the arm base link frame to the end effector frame given
    the current robot config.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]

    Returns:
        4x4 np-array: SE(3) transformation from the arm base link frame to the end effector frame,
                      T0e
    """


    joints = config[3:8]
    return mr.FKinBody(M0e,Jarm_home,joints)

def calculate_Tse(config):
    """
    Calculate the SE(3) transformation from the space frame to the end effector frame given the
    current robot config.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]

    Returns:
        4x4 np-array: SE(3) transformation from the space frame to the end effector frame, Tse
    """

    Ts0 = calculate_Ts0(config)
    T0e = calculate_T0e(config)

    # Tse
    return Ts0 @ T0e

def calculate_Tsc(x,y,theta):
    """
    Calculate the transformation from the space frame to a cube frame (on the ground)
    given it's 2D configuration.

    Args:
        x (float): x position of cube
        y (float): y position of cube
        theta (float): rotation of cube about the z-axis

    Returns:
        4x4 np-array: SE(3) transformation from the space frame to the cube frame, Tsc
    """
    return np.array([
        [np.cos(theta),-np.sin(theta),0,x],
        [np.sin(theta),np.cos(theta),0,y],
        [0,0,1,0.025],
        [0,0,0,1],
    ])

def get_arm_Jacobian(config):
    """
    Calculate the body jacobian of the arm given the robot's current configuration.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]

    Returns:
        6x5 np-array: the current body jacobian for the arm based on the current joint angles
    """

    joints = config[3:8]

    return mr.JacobianBody(Jarm_home,joints)

def get_chassis_Jacobian(config):
    """
    Calculate the body jacobian of the chassis given the robot's current configuration.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]

    Returns:
        6x4 np-array: the current body jacobian for the chassis based on the current config
    """

    F = np.linalg.pinv(CHASSIS_H0)
    m = F.shape[1]
    Zm = np.array([0]*m)
    
    F6 = np.vstack((Zm,Zm,F,Zm))
    T0e = calculate_T0e(config)

    return mr.Adjoint(np.linalg.inv(T0e) @ np.linalg.inv(Tb0)) @ F6

def get_full_Jacobian(config):
    """
    Calculate the full body jacobian of the robot given the robot's current configuration.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]

    Returns:
        6x9 np-array: the full current body jacobian for the robot based on the current config
    """

    Jarm = get_arm_Jacobian(config)
    Jchassis = get_chassis_Jacobian(config)

    return np.hstack((Jchassis, Jarm))

def test_joint_limits(config, joint_limits):
    """
    Test whether the robot's joint limits have been exceeded by a particular configuration.

    Args:
        config (np-array, 13-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper]
        joint_limits (2x5 np-array): limits on the value of each joint. First row is minimum joint
                                     value, second row is maximum joint value

    Returns:
        np-array, 5-vector: boolean array indicating if each joint has exceeded its limit.
                            True if it has, False if it hasn't
    """
    joints = config[3:8]

    lower_limits = joints < joint_limits[0,:]
    upper_limits = joints > joint_limits[1,:]

    return np.logical_or(lower_limits,upper_limits)