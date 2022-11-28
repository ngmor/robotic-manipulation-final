#!/usr/bin/env python3

# To run the example code and generate a trajectory CSV, simply run this python file.
# On a typical Linux installation, from the directory the code is in:
#
# python3 traj_gen.py

import modern_robotics as mr
import numpy as np
from common import calculate_Tse, get_full_Jacobian, ZERO_TOL

def FeedbackControl(Tse_act, Tse_des,Tse_des_next,kp,ki,dt,Xerr_integral_sum):
    """
    Determine commanded end-effector twist by comparing current configuration with reference trajectory.

    Args:
        Tse_act (np-array, 4x4): Actual current transformation from space frame to end-effector frame
        Tse_des (np-array, 4x4): Desired current transformation from space frame to end-effector frame
        Tse_des_next (np-array, 4x4): Desired transformation from space frame to end-effector frame at next time step
        kp (float): proportional gain constant
        ki (float): integral gain constant
        dt (float): timestep (sec)
        Xerr_integral_sum: (np-array, 6-vector): running count of summed integral error
    Returns:
        V: (np-array, 6-vector): Commanded end-effector twist
        Xerr_integral_sum: (np-array, 6-vector): updated summed integral error
    """
    
    # Calculate feedforward reference twist
    Vd_mat = mr.MatrixLog6(np.linalg.inv(Tse_des) @ Tse_des_next) / dt
    Vd = mr.se3ToVec(Vd_mat)
    
    XinvXdes = np.linalg.inv(Tse_act) @ Tse_des
    
    # Calculate error twist
    Xerr_mat = mr.MatrixLog6(XinvXdes)
    Xerr = mr.se3ToVec(Xerr_mat)

    # Add to integral sum
    Xerr_integral_sum += Xerr * dt

    # Generate gain matrices
    I6 = np.eye(len(Xerr),len(Xerr))
    Kp = kp * I6
    Ki = ki * I6 
    
    # Calculate commanded end effector twist
    V = mr.Adjoint(XinvXdes) @ Vd + Kp @ Xerr + Ki @ Xerr_integral_sum
    
    return V, Xerr_integral_sum

def get_velocities_from_twist(twist_cmd,config):
    """
    Translate twist command for end effector into wheel / joint velocities

    Args:
        twist_cmd (np-array, 6-vector): Commanded end-effector twist
        config (np-array, 12-vector): current config of robot, in this order:
            [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4]

    Returns:
        velocity_cmd (np-array, 9-vector): Commanded wheel/joint velocities:
            [W1d, W2d, W3d, W4d, J1d, J2d, J3d, J4d, J5d]
    """

    Je = get_full_Jacobian(config)

    return np.linalg.pinv(Je,rcond=ZERO_TOL) @ twist_cmd

if __name__ == "__main__":

    current_position = np.array([0,0,0,0,0,0.2,-1.6,0])

    Tse_act = calculate_Tse(current_position)

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

    kp = 1
    ki = 0
    dt = 0.01
    Xerr_integral_sum = np.array([0.]*6)

    [V_cmd, Xerr_integral_sum] = FeedbackControl(Tse_act, Tse_des,Tse_des_next,kp,ki,dt,Xerr_integral_sum)

    velocity_cmd = get_velocities_from_twist(V_cmd,current_position)

    # TODO - implement self collisions at this step?