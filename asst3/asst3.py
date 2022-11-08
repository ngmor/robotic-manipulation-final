#!/usr/bin/env python3

import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
import os

def Puppet(thetalist, dthetalist, g, Mlist, Slist,Glist,t,dt,
                damping,stiffness,springPos,restLength):
    """Puppet

    :param thetalist: (rad) n-vector of initial joint angles
    :param dthetalist: (rad/s) n-vector of initial joint rates
    :param g: (m/s^2) gravity 3-vector
    :param Mlist: the configurations of the link frames relative to each other
                  at the home configuration (list of transformation matrices)
    :param Slist: the screw axes in the space frame when the robot is at its
                  home configuration
    :param Glist: (kg/kgm^2) the spatial inertia matrices Gi of the links
    :param t: (sec) total simulation time
    :param dt: (sec) simulation timestep
    :param damping: (Nms/rad) scalar indicating viscous damping at each joint
    :param stiffness: (N/m) scalar indicating the stiffness of the spring
    :param springPos: (m) 3-vector indicating the location of the end of the spring
                      not attached to the robot, expressed in the {s} frame
    :param restLength: (m) scalar indicating the length of the spring when
                       it is at rest

    :return [thetamat, dthetamat]:
        thetamat: N x n matrix where row i is the set of joint values after
                  simulation step i - 1
        dthetamat: N x n matrix where row i is the set of joint rates after
                  simulation step i - 1

    """
    # n = number of joints
    n = thetalist.shape[0]
    
    # N = number of timesteps
    N = int(np.ceil(t / dt))
    
    # init Ftip in case we don't calculate it
    Ftip_ee = np.array([0.]*6)

    # init looping variables
    theta = np.copy(thetalist)
    dtheta = np.copy(dthetalist)
    
    thetamat = np.zeros((N,n))
    dthetamat = np.zeros((N,n))

    # calculate M for entire robot
    M = np.eye(4,4)
    for tf in Mlist:
        M = np.matmul(M,tf)

    # iterate through all timesteps
    for i in range(0,N):
        # Store theta and dtheta values
        thetamat[i,:] = theta.copy()
        dthetamat[i,:] = dtheta.copy()

        # calculate damping joint torques
        tau = -damping*dtheta

        # only do spring calculate if stiffness is not 0
        # to save calculation time
        if stiffness > 0.00001:
            
            # calculate forward kinematics
            Tsb = mr.FKinSpace(M,Slist,theta)

            #Get end effector coordinates
            ee_coord = Tsb[0:3,3]
            
            # Calculate spring vector in space frame
            spring_vec = springPos - ee_coord

            # Get spring length and unit vector
            length = np.linalg.norm(spring_vec)
            spring_unit_vec = spring_vec / length
            
            # Calculate force of the tip in space frame
            # Negative sign because we need the force the end effector
            # applies, and the positive sign is the force the spring applies
            # to the end effector
            F_space = -stiffness*(length - restLength)*spring_unit_vec

            # Calculate moment about spatial frame caused by force
            m_space = np.cross(ee_coord,F_space)

            # Compose wrench in space frame
            Ftip_space = np.hstack([m_space,F_space])
            
            # Transform wrench to body frame
            Ftip_ee = np.matmul(mr.Adjoint(Tsb).T, Ftip_space)

        # calculate forward dynamics
        ddtheta = mr.ForwardDynamics(theta,dtheta,tau,g,Ftip_ee,Mlist,Glist,Slist)

        # Euler integration
        [theta,dtheta] = mr.EulerStep(theta,dtheta,ddtheta,dt)

        # Check for validity
        if any(np.isnan(theta)) or any(np.isnan(dtheta)):
            print(f'Error: invalid values in at {dt*i:.3f} sec.')
            break


    return [thetamat,dthetamat]

def generate_csv(folder,filename,thetamat):
    if folder != '':
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder += '/'

    filepath = folder + filename
    np.savetxt(filepath,thetamat,delimiter=',',fmt='%10.6f')

# UR5 parameters
M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
Glist = [G1, G2, G3, G4, G5, G6]
Mlist = [M01, M12, M23, M34, M45, M56, M67] 
Slist = [[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]]

# Other inputs
thetalist = np.array([0.]*6)
dthetalist = np.array([0.]*6)
springPos = np.array([0,0,2])
restLength = 0.0

# File generation
folder = 'csv'

# Part 1A ---------------------------------------------------------------------------
run_part_1A = True

filename = 'part1A.csv'

# part specific inputs
t = 5 # sec
dt = 0.001 # sec
g = np.array([0,0,-9.81])
damping = 0.0
stiffness = 0.0

if run_part_1A:
    print('Start Part 1A')

    # function call
    [thetamat,dthetamat] = Puppet(thetalist, dthetalist, g, Mlist, Slist,Glist,t,dt,
                                    damping,stiffness,springPos,restLength)

    generate_csv(folder,filename,thetamat)
    print('Part 1A Complete')

# Part 1B ---------------------------------------------------------------------------
run_part_1B = True

filename = 'part1B.csv'

# part specific inputs
t = 5 # sec
dt = 0.05 # sec
g = np.array([0,0,-9.81])
damping = 0.0
stiffness = 0.0

if run_part_1B:
    print('Start Part 1B')

    # function call
    [thetamat,dthetamat] = Puppet(thetalist, dthetalist, g, Mlist, Slist,Glist,t,dt,
                                    damping,stiffness,springPos,restLength)

    generate_csv(folder,filename,thetamat)
    print('Part 1B Complete')

# Part 2A ---------------------------------------------------------------------------
run_part_2A = True

filename = 'part2A.csv'

# part specific inputs
t = 5 # sec
dt = 0.01 # sec
g = np.array([0,0,-9.81])
damping = 3.0
stiffness = 0.0

if run_part_2A:
    print('Start Part 2A')

    # function call
    [thetamat,dthetamat] = Puppet(thetalist, dthetalist, g, Mlist, Slist,Glist,t,dt,
                                    damping,stiffness,springPos,restLength)

    generate_csv(folder,filename,thetamat)
    print('Part 2A Complete')
    

# Part 2B ---------------------------------------------------------------------------
run_part_2B = True

filename = 'part2B.csv'

# part specific inputs
t = 5 # sec
dt = 0.01 # sec
g = np.array([0,0,-9.81])
damping = -0.025
stiffness = 0.0

if run_part_2B:
    print('Start Part 2B')

    # function call
    [thetamat,dthetamat] = Puppet(thetalist, dthetalist, g, Mlist, Slist,Glist,t,dt,
                                    damping,stiffness,springPos,restLength)

    generate_csv(folder,filename,thetamat)
    print('Part 2B Complete')

# Part 3A ---------------------------------------------------------------------------
run_part_3A = True

filename = 'part3A.csv'

# part specific inputs
t = 10 # sec
dt = 0.01 # sec
g = np.array([0,0,0])
damping = 0.0
stiffness = 2.0

if run_part_3A:
    print('Start Part 3A')

    # function call
    [thetamat,dthetamat] = Puppet(thetalist, dthetalist, g, Mlist, Slist,Glist,t,dt,
                                    damping,stiffness,springPos,restLength)

    generate_csv(folder,filename,thetamat)
    print('Part 3A Complete')

# Part 3B ---------------------------------------------------------------------------
run_part_3B = True

filename = 'part3B.csv'

# part specific inputs
t = 10 # sec
dt = 0.01 # sec
g = np.array([0,0,0])
damping = 1.5
stiffness = 2.0

if run_part_3B:
    print('Start Part 3B')

    # function call
    [thetamat,dthetamat] = Puppet(thetalist, dthetalist, g, Mlist, Slist,Glist,t,dt,
                                    damping,stiffness,springPos,restLength)

    generate_csv(folder,filename,thetamat)
    print('Part 3B Complete')