#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
import os



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
    :return theta_iterations: The thetalist at each performed iteration
    :return err_lin_iterations: The linear error at each performed iteration
    :return err_ang_iterations: The angular error at each performed iteration
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

    # Split out some of these calculations so we can print them
    Tsb = mr.FKinBody(M, Blist, thetalist)
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
    err_ang = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    err_lin = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err = err_ang > eomg or err_lin > ev

    # Store and print info
    theta_iterations = [thetalist]
    err_lin_iterations = [err_lin]
    err_ang_iterations = [err_ang]
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
        # Bound thetas between 2pi and -2pi
        # Might not do this in every case, but it works for this robot,
        # which only has revolute joints with these bounds
        for j, theta in enumerate(thetalist):
            while not ((thetalist[j] < 2*np.pi) and (thetalist[j] > -2*np.pi)):
                if thetalist[j] >= 2*np.pi:
                    thetalist[j] -= 2*np.pi
                elif thetalist[j] <= -2*np.pi:
                    thetalist[j] += 2*np.pi

        i = i + 1

        # Split out some of these calculations so we can print them
        Tsb = mr.FKinBody(M, Blist, thetalist)
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
        err_ang = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        err_lin = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        err = err_ang > eomg or err_lin > ev

        # Store and print info
        theta_iterations.append(thetalist)
        err_lin_iterations.append(err_lin)
        err_ang_iterations.append(err_ang)
        print(f'\n\n\nIteration {i}:\n')
        print(f'Joint vector:\n{thetalist}\n')
        print(f'SE(3) end-effector config:\n{Tsb}\n')
        print(f'Error twist V_b:\t\t{Vb}')
        print(f'Angular error ||omega_b||:\t{err_ang}')
        print(f'Linear error ||v_b||:\t\t{err_lin}')

    theta_iterations = np.array(theta_iterations)
    
    # Round to 6 decimal places
    theta_iterations = np.round(theta_iterations, 6)

    # Output to CSV
    np.savetxt(filename,theta_iterations, delimiter=',',fmt='%10.6f')


    # Return thetalist and success status, but also iteration data so it can be plotted
    return (thetalist, not err, theta_iterations,err_lin_iterations,err_ang_iterations)



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

# File saving
folder = 'csv'
if folder != '':
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder += '/'

short_file = folder + 'short_iterates.csv'
long_file = folder + 'long_iterates.csv'

# Initial guess for joint angles for short iteration
theta0_short = np.array([1.763,0,-1.825,1.0,1.578,0])

# Call function for short iteration
print('\n\n\n--------------- Short Iterations:')
[theta_sol_short, success_short, theta_iter_short, err_lin_iter_short, err_ang_iter_short] = \
    IKinBodyIterates(Jb_Home, M, Tsb_desired, theta0_short, tol_ang, tol_lin,short_file)

if success_short:
    print('\n\nConverged')
else:
    print('\n\nDid not converge')

# Initial guess for joint angles for long iteration
theta0_long = np.array([1.0,0,-1.0,0.6,0.5,-0.2])

# Call function for long iteration
print('\n\n\n--------------- Long Iterations:')
[theta_sol_long, success_long, theta_iter_long, err_lin_iter_long, err_ang_iter_long] = \
    IKinBodyIterates(Jb_Home, M, Tsb_desired, theta0_long, tol_ang, tol_lin,long_file)

if success_long:
    print('\n\nConverged')
else:
    print('\n\nDid not converge')


# Plots

# End effector position
# Generate end effector position data using FKinBody
# and output thetalist iterations
end_effector_x_short = []
end_effector_y_short = []
end_effector_z_short = []
for i, theta in enumerate(theta_iter_short):
    Tsb = mr.FKinBody(M,Jb_Home,theta)
    end_effector_x_short.append(Tsb[0,3])
    end_effector_y_short.append(Tsb[1,3])
    end_effector_z_short.append(Tsb[2,3])


end_effector_x_long = []
end_effector_y_long = []
end_effector_z_long = []
for theta in theta_iter_long:
    Tsb = mr.FKinBody(M,Jb_Home,theta)
    end_effector_x_long.append(Tsb[0,3])
    end_effector_y_long.append(Tsb[1,3])
    end_effector_z_long.append(Tsb[2,3])

# Get start and end points
start_x = [end_effector_x_short[0],end_effector_x_long[0]]
start_y = [end_effector_y_short[0],end_effector_y_long[0]]
start_z = [end_effector_z_short[0],end_effector_z_long[0]]

end_x = [end_effector_x_short[-1],end_effector_x_long[-1]]
end_y = [end_effector_y_short[-1],end_effector_y_long[-1]]
end_z = [end_effector_z_short[-1],end_effector_z_long[-1]]

fig_3d = plt.figure('End effector position from initial guess to convergence')
ax3d = plt.axes(projection='3d')
ax3d.plot(end_effector_x_short, end_effector_y_short, end_effector_z_short,label='short')
ax3d.plot(end_effector_x_long, end_effector_y_long, end_effector_z_long,label='long')
ax3d.scatter(start_x, start_y, start_z,label='start',c='black',marker='o')
ax3d.scatter(end_x, end_y, end_z,label='end',c='black',marker='x')
ax3d.set_xlabel('x (m)')
ax3d.set_ylabel('y (m)')
ax3d.set_zlabel('z (m)')
ax3d.set_title('End effector position from initial guess to convergence')
ax3d.legend()


# linear error
fig_err_lin = plt.figure('Linear error over iterations')

ax_err_lin = fig_err_lin.add_subplot()
ax_err_lin.set_xlabel('Iteration')
ax_err_lin.set_ylabel('Linear error (m)')
ax_err_lin.set_title('Linear error over iterations')

ax_err_lin.plot(np.arange(len(err_lin_iter_short)), err_lin_iter_short,label='short')
ax_err_lin.plot(np.arange(len(err_lin_iter_long)), err_lin_iter_long,label='long')
ax_err_lin.legend()

# Angular error
fig_err_ang = plt.figure('Angular error over iterations')

ax_err_ang = fig_err_ang.add_subplot()
ax_err_ang.set_xlabel('Iteration')
ax_err_ang.set_ylabel('Angular error (rad)')
ax_err_ang.set_title('Angular error over iterations')

ax_err_ang.plot(np.arange(len(err_ang_iter_short)), err_ang_iter_short,label='short')
ax_err_ang.plot(np.arange(len(err_ang_iter_long)), err_ang_iter_long,label='long')
ax_err_ang.legend()


plt.show()