#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np



def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev, filename):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :param filename: name of csv file where theta matrix will be stored
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = mr.FKinBody(M, Blist, thetalist)
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
    err_ang = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    err_lin = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err = err_ang > eomg or err_lin > ev

    theta_iterations = [thetalist]
    print(f'Iteration {i}:\n')
    print(f'Joint vector:\n{thetalist}\n')
    print(f'SE(3) end-effector config:\n{Tsb}\n')
    print(f'Error twist V_b:\t\t{Vb}')
    print(f'Angular error ||omega_b||:\t{err_ang}')
    print(f'Linear error ||v_b||:\t\t{err_lin}')

    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, \
                                                         thetalist)), Vb)
        i = i + 1
        Tsb = mr.FKinBody(M, Blist, thetalist)
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
        err_ang = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        err_lin = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        err = err_ang > eomg or err_lin > ev

        theta_iterations.append(thetalist)
        print(f'\n\n\nIteration {i}:\n')
        print(f'Joint vector:\n{thetalist}\n')
        print(f'SE(3) end-effector config:\n{Tsb}\n')
        print(f'Error twist V_b:\t\t{Vb}')
        print(f'Angular error ||omega_b||:\t{err_ang}')
        print(f'Linear error ||v_b||:\t\t{err_lin}')

    print('\n\n\n')
    theta_iterations = np.array(theta_iterations)
    
    # Round to 6 decimal places
    theta_iterations = np.round(theta_iterations, 6)

    # Output to CSV
    np.savetxt(filename,theta_iterations, delimiter=',',fmt='%10.6f')

    return (thetalist, not err)



# Robot size parameters (mm)
W1 = 109 / 1000
W2 = 82 / 1000
L1 = 425 / 1000
L2 = 392 / 1000
H1 = 89 / 1000
H2 = 95 / 1000

# Body Jacobian at home position
Jb_Home = np.array([
    [0, 1, 0, W1 + W2, 0, L1 + L2],
    [0, 0, 1, H2, -L1 - L2, 0],
    [0, 0, 1, H2, -L2, 0],
    [0, 0, 1, H2, 0, 0],
    [0, -1, 0, -W2, 0, 0],
    [0, 0, 1, 0, 0, 0],
]).T

# Home configuration of end effector
M = np.array([
    [-1, 0, 0, L1 + L2],
    [0, 0, 1, W1 + W2],
    [0, 1, 0, H1 - H2],
    [0, 0, 0, 1],
])

# Desired configuration of the end effector
Tsb_desired = np.array([
    [-1, 0, 0, -0.2],
    [0, 0, 1, 0.6],
    [0, 1, 0, 0.35],
    [0, 0, 0, 1],
])

# Angular and linear tolerances
tol_ang = 0.001 # rad
tol_lin = 0.0001 # m

csv_file = 'asst2_csvs/test.csv'

# Initial guess for joint angles
theta0 = np.array([0,0,0,0,0,0])

[thetai, success] = IKinBodyIterates(Jb_Home, M, Tsb_desired, theta0, tol_ang, tol_lin,csv_file)

if success:
    print('\nConverged')
else:
    print('\nDid not converge')