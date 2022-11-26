#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
from traj_gen import TrajectoryGenerator
from common import generate_csv

if __name__ == "__main__":

    # initial end effector transform
    Tse_ini = np.array([
        [0,0,1,0],
        [0,1,0,0],
        [-1,0,0,0.5],
        [0,0,0,1]
    ])

    # initial cube transform
    Tsc_ini = np.array([
        [1,0,0,1],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])

    # final cube transform
    Tsc_fin = np.array([
        [0,1,0,0],
        [-1,0,0,-1],
        [0,0,1,0],
        [0,0,0,1]
    ])

    # grasp transform
    # Rotated around the y axis from the cube position by the specified angle
    # Translated in the positive z axis from the cube position by the specified height
    grasp_angle = 3*np.pi/4 # rad
    grasp_height = 0.025 # m
    Tce_grasp = np.array([
        [np.cos(grasp_angle),0,np.sin(grasp_angle),0],
        [0,1,0,0],
        [-np.sin(grasp_angle),0,np.cos(grasp_angle),0.025],
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
    

    # define timing information
    dt = 0.01 # sec
    total_time = 20. # sec
    gripper_actuate_time = 0.625 # sec
    standoff_time = 1.5 # sec
    k = 1

    # generate trajectory
    traj = TrajectoryGenerator(Tse_ini,Tsc_ini,Tsc_fin,Tce_grasp,Tce_standoff,k,
                               dt,total_time,gripper_actuate_time,standoff_time)

    # Save to CSV
    generate_csv('traj.csv',traj,folder='traj')