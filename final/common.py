#!/usr/bin/env python3

import numpy as np
import os

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

chassis_H0 = _generate_chassis_H0()