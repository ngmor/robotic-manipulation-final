#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
import os

# TODO - move into function? should these be global?
dt = 0.01
total_time = 30. # sec
gripper_actuate_time = 0.625 # sec
standoff_time = 1.5 # sec

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

# grasp
grasp_angle = 3*np.pi/4
Tce_grasp = np.array([
    [np.cos(grasp_angle),0,np.sin(grasp_angle),0],
    [0,1,0,0],
    [-np.sin(grasp_angle),0,np.cos(grasp_angle),0],
    [0,0,0,1]
])

# grasp to standoff
# Tgrasp_standoff @ Tce_grasp
standoff_height = 0.075
Tgrasp_standoff = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,standoff_height],
    [0,0,0,1]
])


def TrajectoryGenerator(Tse_ini,Tsc_ini,Tsc_fin,Tce_grasp,Tce_standoff,k):
    # mr.CartesianTrajectory(Xstart, Xend, Tf, N, 5)
    # mr.ScrewTrajectory(Xstart, Xend, Tf, N, 5)

    # get total number of timesteps
    N = int(np.ceil(total_time*k / dt))
    method = 5

    # Divide each segment's timesteps
    N2 = int(np.ceil(standoff_time)*k / dt)
    N4 = N2
    N6 = N2
    N8 = N2
    N3 = int(np.ceil((1.2 * gripper_actuate_time)*k / dt))
    N7 = N3
    N1 = int(np.ceil((N - (N2 + N3 + N4 + N6 + N7 + N8)) / 2.))
    N5 = N - (N1 + N2 + N3 + N4 + N6 + N7 + N8)

    # print(f'N: {N}')
    # print(f'N1: {N1}')
    # print(f'N2: {N2}')
    # print(f'N3: {N3}')
    # print(f'N4: {N4}')
    # print(f'N5: {N5}')
    # print(f'N6: {N6}')
    # print(f'N7: {N7}')
    # print(f'N8: {N8}')
    # print(f'Total: {N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8}')

    # Init trajectory
    traj = np.zeros((N,13))
    offset = 0

    # Segment 1:
    # Gripper to pick standoff configuration above block
    seg1 = mr.ScrewTrajectory(Tse_ini, Tsc_ini @ Tce_standoff, (N1 - 1)*dt, N1, method)

    for i in range(len(seg1)):
        traj[offset + i] = components_to_csv_line(seg1[i],0)
    
    offset += i + 1

    # Segment 2:
    # Gripper down to pick position

    # Segment 3:
    # Close gripper

    # Segment 4:
    # Move gripper back up to pick standoff

    # Segment 5:
    # Move gripper to place standoff

    # Segment 6:
    # Gripper down to place position

    # Segment 7:
    # Open gripper

    # Segment 8:
    # Move gripper back up to place standoff


    return traj

def components_to_csv_line(Tse,gripper_state):
    return np.array([
        Tse[0][0],
        Tse[0][1],
        Tse[0][2],
        Tse[1][0],
        Tse[1][1],
        Tse[1][2],
        Tse[2][0],
        Tse[2][1],
        Tse[2][2],
        Tse[0][3],
        Tse[1][3],
        Tse[2][3],
        gripper_state
    ])

def generate_csv(folder,filename,traj):
    if folder != '':
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder += '/'

    filepath = folder + filename
    np.savetxt(filepath,traj,delimiter=',',fmt='%10.6f')

if __name__ == "__main__":
    Tce_standoff = Tgrasp_standoff @ Tce_grasp
    traj = TrajectoryGenerator(Tse_ini,Tsc_ini,Tsc_fin,Tce_grasp,Tce_standoff,1)

    generate_csv('traj','traj.csv',traj)