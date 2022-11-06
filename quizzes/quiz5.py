#!/usr/bin/env python3

import numpy as np
import modern_robotics as mr

#1
Fs = np.array([[0,0,0,2,0,0]]).T

JsT = np.array([[0,0,1,0,0,0],
               [0,0,1,0,-1,0],
               [0,0,1,np.sin(np.pi/4),-(1+np.cos(np.pi/4)),0]])

T = np.matmul(JsT,Fs)
print("1:")
print(T)

#2
Jb = np.array([[1,1,1,1],
              [-1,-1,-1,0],
              [3,2,1,1]])
Fb = np.array([[10,10,10]]).T

T = np.matmul(Jb.T,Fb)
print("2:")
print(T)

#3
Slist = np.array([[0,0,1,0,0,0],
                  [1,0,0,0,2,0],
                  [0,0,0,0,1,0]]).T
thetalist = [np.pi/2,np.pi/2,1]
Js = mr.JacobianSpace(Slist,thetalist)

print("3:")
print(Js)

#4
Slist = np.array([[0,1,0,3,0,0],
                  [-1,0,0,0,3,0],
                  [0,0,0,0,0,1]]).T
thetalist = [np.pi/2,np.pi/2,1]
Jb = mr.JacobianBody(Slist,thetalist)

print("4:")
print(Jb)

#5
Jbv = np.array([[-0.105,0,0.006,-0.045,0,0.006,0],
               [-0.889,0.006,0,-0.844,0.006,0,0],
               [0,-0.105,0.889,0,0,0,0]])

print('5:')
A = np.matmul(Jbv,Jbv.T)
[lam,v] = np.linalg.eig(A)
print(lam)
print(v)

print(np.sqrt(lam[1]))