#!/usr/bin/env python3

import numpy as np
import modern_robotics as mr

I = np.eye(3,3)

#9
s_theta = np.array([[0,1,2,3,0,0]]).T

w_theta = s_theta[:3]

w_theta_mag = np.linalg.norm(w_theta)
theta = w_theta_mag
print(f'theta = {theta}')
s = s_theta / theta
print(f's = {s}')
w = s[:3]
v = s[3:]

w_mat = np.array([[0,-w[2][0],w[1][0]]
                 ,[w[2][0],0,-w[0][0]]
                 ,[-w[1][0],w[0][0],0]])

R = I + np.sin(theta)*w_mat + (1 - np.cos(theta))*np.matmul(w_mat,w_mat)

other = np.matmul(I*theta + (1 - np.cos(theta))*w_mat + (theta - np.sin(theta))*np.matmul(w_mat,w_mat),v)
print(R)
print(other)

#11
T = np.array([[0,-1,0,3]
             ,[1,0,0,0]
             ,[0,0,1,1]
             ,[0,0,0,1]])

T_inv = mr.TransInv(T)
print(T_inv)

#12
twist = np.array([1,0,0,0,2,3])
twist_mat = mr.VecTose3(twist)
print(twist_mat)

#13
screw = mr.ScrewToAxis([0,0,2],[1,0,0],1)
print(screw)

#14
exp = np.array([[0,-1.5708,0,2.3562]
               ,[1.5708,0,0,-2.3562]
               ,[0,0,0,1]
               ,[0,0,0,0]])

T = mr.MatrixExp6(exp)
print(T)

#15
T = np.array([[0,-1,0,3],[1,0,0,0],[0,0,1,1],[0,0,0,1]])
exp = mr.MatrixLog6(T)
print(exp)
