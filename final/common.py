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
    """TODO"""
    return (1 / (r*np.cos(gamma))) * np.array([
        x*np.sin(beta + gamma) - y*np.cos(beta + gamma),
        np.cos(beta + gamma + phi),
        np.sin(beta + gamma + phi),
    ]).T

def generate_H0(wheels):
    """TODO"""
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

def calculate_Tsb(phi,x,y):
    """TODO"""
    return np.array([
        [np.cos(phi), -np.sin(phi), 0, x],
        [np.sin(phi), np.cos(phi), 0, y],
        [0,0,1,0.0963],
        [0,0,0,1],
    ])

def calculate_Ts0(phi,x,y):
    """TODO"""
    Tsb = calculate_Tsb(phi,x,y)
    return Tsb @ Tb0

def youbot_FK(current_config):
    """TODO"""
    
    phi = current_config[0]
    x = current_config[1]
    y = current_config[2]
    joints = current_config[3:9]

    Ts0 = calculate_Ts0(phi,x,y)
    T0e = mr.FKinBody(M0e,Jarm_home,joints)

    # Tse
    return Ts0 @ T0e

