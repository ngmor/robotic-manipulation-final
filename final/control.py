#!/usr/bin/env python3

# To run the example code and generate a trajectory CSV, simply run this python file.
# On a typical Linux installation, from the directory the code is in:
#
# python3 traj_gen.py

import modern_robotics as mr
import numpy as np
from common import youbot_FK

def FeedbackControl(Tse_act, Tse_des,Tse_des_next,kp,ki,dt):
    """
    Determine commanded end-effector twist by comparing current configuration with reference trajectory.

    Args:
        Tse_act (np-array, 4x4): Actual current transformation from space frame to end-effector frame
        Tse_des (np-array, 4x4): Desired current transformation from space frame to end-effector frame
        Tse_des_next (np-array, 4x4): Desired transformation from space frame to end-effector frame at next time step
        kp (float): proportional gain constant
        ki (float): integral gain constant
        dt (float): timestep (sec)
    """    
    
    pass


if __name__ == "__main__":

    current_position = np.array([0,0,0,0,0,0.2,-1.6,0])

    Tse_act = youbot_FK(current_position)

    # Testing code
    Tse_des = np.array([
        [0,0,1,0.5],
        [0,1,0,0],
        [-1,0,0,0.5],
        [0,0,0,1]
    ])

    Tse_des_next = np.array([
        [0,0,1,0.6],
        [0,1,0,0],
        [-1,0,0,0.3],
        [0,0,0,1]
    ])

    kp = 0
    ki = 0
    dt = 0.01

    FeedbackControl(Tse_act, Tse_des,Tse_des_next,kp,ki,dt)