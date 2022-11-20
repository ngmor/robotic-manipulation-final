#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
import os

def TrajectoryGenerator(Tse_ini,Tsc_ini,Tsc_fin,Tce_grasp,Tce_standoff,k,
                        dt,total_time,gripper_actuate_time,standoff_time):
    """
    Generates a trajectory for the end effector of the robot for the final project.

    Args:
        Tse_ini (4x4 np array): SE(3) transform from space to end effector at its initial location
        Tsc_ini (4x4 np array): SE(3) transform from space to cube at it's initial location
        Tsc_fin (4x4 np array): SE(3) transform from space to cube at it's final location
        Tce_grasp (4x4 np array): SE(3) transform from cube to end effector at grasp position
        Tce_standoff (4x4 np array): SE(3) transform from cube to end effector at standoff position
        k (int): number of trajectory reference configurations per dt seconds
        dt (float): timestep in seconds
        total_time (float): total trajectory time in seconds
        gripper_actuate_time (float): time for the gripper to actuate in seconds
        standoff_time (float): time to move from the standoff position to the grasp position in seconds

    Returns:
        n x 13 np array: Generated trajectory
    """    

    # get total number of timesteps
    N = int(np.ceil(total_time*k / dt))
    method = 5

    # Divide each segment's timesteps
    # based on input times for each segment
    N2 = int(np.ceil(standoff_time)*k / dt)
    N4 = N2
    N6 = N2
    N8 = N2
    N3 = int(np.ceil((1.2 * gripper_actuate_time)*k / dt))
    N7 = N3
    N1 = int(np.ceil((N - (N2 + N3 + N4 + N6 + N7 + N8)) / 2.))
    N5 = N - (N1 + N2 + N3 + N4 + N6 + N7 + N8)

    # Init trajectory matrix to hold output of all segments
    traj = np.zeros((N,13))
    offset = 0

    # Segment 1:
    # Gripper to pick standoff configuration above block
    # Calculate pick standoff position transform in space frame
    Tse_pick_standoff = Tsc_ini @ Tce_standoff

    # Generate trajectory
    seg1 = mr.ScrewTrajectory(Tse_ini, Tse_pick_standoff, (N1 - 1)*dt, N1, method)

    # Store in trajectory matrix
    for i in range(len(seg1)):
        traj[offset + i] = components_to_csv_line(seg1[i],0)
    
    offset += i + 1

    # Segment 2:
    # Gripper down to pick position
    # Calculate pick position transform in space frame
    Tse_pick = Tsc_ini @ Tce_grasp

    # Generate trajectory
    seg2 = mr.CartesianTrajectory(Tse_pick_standoff, Tse_pick, (N2 - 1)*dt, N2, method)

    # Store in trajectory matrix
    for i in range(len(seg2)):
        traj[offset + i] = components_to_csv_line(seg2[i],0)
    
    offset += i + 1

    # Segment 3:
    # Close gripper
    # keep current end effector position transform, just command the gripper close with a "1"
    for i in range(N3):
        traj[offset + i] = components_to_csv_line(Tse_pick,1)

    offset += i + 1

    # Segment 4:
    # Move gripper back up to pick standoff
    
    # Generate trajectory
    seg4 = mr.CartesianTrajectory(Tse_pick, Tse_pick_standoff, (N4 - 1)*dt, N4, method)

    # Store in trajectory matrix
    for i in range(len(seg4)):
        traj[offset + i] = components_to_csv_line(seg4[i],1)
    
    offset += i + 1

    # Segment 5:
    # Move gripper to place standoff

    # Calculate place standoff position transform in space frame
    Tse_place_standoff = Tsc_fin @ Tce_standoff
    
    # Generate trajectory
    seg5 = mr.ScrewTrajectory(Tse_pick_standoff, Tse_place_standoff, (N5 - 1)*dt, N5, method)

    # Store in trajectory matrix
    for i in range(len(seg5)):
        traj[offset + i] = components_to_csv_line(seg5[i],1)
    
    offset += i + 1

    # Segment 6:
    # Gripper down to place position

    # Calculate place position transform in space frame
    Tse_place = Tsc_fin @ Tce_grasp

    # Generate trajectory
    seg6 = mr.CartesianTrajectory(Tse_place_standoff, Tse_place, (N6 - 1)*dt, N6, method)

    # Store in trajectory matrix
    for i in range(len(seg6)):
        traj[offset + i] = components_to_csv_line(seg6[i],1)
    
    offset += i + 1

    # Segment 7:
    # Open gripper
    # keep current end effector position transform, just command the gripper open with a "0"
    for i in range(N7):
        traj[offset + i] = components_to_csv_line(Tse_place,0)

    offset += i + 1

    # Segment 8:
    # Move gripper back up to place standoff

    # Generate trajectory
    seg8 = mr.CartesianTrajectory(Tse_place, Tse_place_standoff, (N8 - 1)*dt, N8, method)

    # Store in trajectory matrix
    for i in range(len(seg8)):
        traj[offset + i] = components_to_csv_line(seg8[i],0)

    # Return trajectory matrix
    return traj


def components_to_csv_line(Tse, gripper_command):
    """
    Convert input space to end effector transform and gripper command to a row in the output
    trajectory matrix/CSV file.

    Args:
        Tse (4x4 np array): SE(3) transformation matrix between space frame and end effector frame
        gripper_command (int): 0 = gripper open, 1 = gripper close

    Returns:
        (1 x 13 np array): row for output CSV
    """

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
        gripper_command
    ])


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