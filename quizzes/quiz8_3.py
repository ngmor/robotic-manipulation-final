#!/usr/bin/env python3

import numpy as np
import modern_robotics as mr

#1
def calc_rotational_inertia_point_masses(m_list,coord_list):
    Ixx = 0
    Ixy = 0
    Ixz = 0
    Iyy = 0
    Iyz = 0
    Izz = 0

    # TODO - error checking
    for i in range(0,len(m_list)):
        x = coord_list[i][0]
        y = coord_list[i][1]
        z = coord_list[i][2]
        m = m_list[i]

        Ixx += m*(y**2 + z**2)
        Ixy += -m*x*y
        Ixz += -m*x*z
        Iyy += m*(x**2 + z**2)
        Iyz += -m*y*z
        Izz += m*(x**2 + y**2)

    return np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])

def calculate_rotational_inertia_rigid_bodies(m_list, coord_list, I_list):
    I = 0
    ident = np.eye(3,3)

    for i in range(0,len(I_list)):
        q = np.array([coord_list[i]]).T
        # Steiner's Theorem
        I += I_list[i] + m_list[i]*(np.matmul(q.T,q)*ident - np.matmul(q,q.T))

    return I

rho = 5600 #kg/m^3

# cylinder
rc = 4 / 2 / 100 #m
hc = 20 / 100 #m

mc = rho*np.pi*rc**2*hc #kg
coordc = [0,0,0]

Ixxc = mc*(3*rc**2 + hc**2) / 12
Iyyc = mc*(3*rc**2 + hc**2) / 12
Izzc = (mc*rc**2) / 2


Ic = np.array([
    [Ixxc,0,0],
    [0,Iyyc,0],
    [0,0,Izzc],
])

# spheres
rs = 20 / 2 / 100 #m

ms = rho*(4/3)*np.pi*rs**3
coords1 = [0,0,hc/2 + rs]
coords2 = [0,0,-(hc/2 + rs)]

Ixxs = (2*ms*rs**2) / 5
Iyys = Ixxs
Izzs = Ixxs

Is = np.array([
    [Ixxs,0,0],
    [0,Iyys,0],
    [0,0,Izzs],
])

# Steiner's theorem
m_list = [mc,ms,ms]
coord_list = [coordc,coords1,coords2]
I_list = [Ic,Is,Is]

print(calculate_rotational_inertia_rigid_bodies(m_list, coord_list, I_list))

ident = np.eye(3,3)
q = np.array([coordc]).T
Icb = Ic + mc*(np.matmul(q.T,q)*ident - np.matmul(q,q.T))

q = np.array([coords1]).T
Is1b = Is + ms*(np.matmul(q.T,q)*ident - np.matmul(q,q.T))

q = np.array([coords2]).T
Is2b = Is + ms*(np.matmul(q.T,q)*ident - np.matmul(q,q.T))

I = Icb + Is1b + Is2b
print(I)


#2
print('\n\n\n\n\n\n')

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

thetalist = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3]
thetadlist = [0.2]*6
thetaddlist = [0.1]*6
g = [0,0,-9.81]
Ftip = [0.1]*6
Mlist = [M01,M12,M23,M34,M45,M56,M67]
Glist = [G1,G2,G3,G4,G5,G6]

print(mr.InverseDynamics(thetalist,thetadlist,thetaddlist,g,Ftip,Mlist,Glist,Slist))