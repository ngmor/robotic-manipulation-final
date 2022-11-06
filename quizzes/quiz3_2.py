#!/usr/bin/env python3

import numpy as np
import modern_robotics as mr

I = np.eye(3,3)

w = np.array([[0,0,2],[0,0,-1],[-2,1,0]])

#R = I

#for i in range(1,2):
#    R += (1/np.math.factorial(i))*w**i

R = mr.MatrixExp3(w)
print(R)

w = mr.VecToso3(np.array([1,2,0.5]))
print(w)

w = np.array([[0,0.5,-1],[-0.5,0,2],[1,-2,0]])
R = mr.MatrixExp3(w)
print(R)

R = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
w = mr.MatrixLog3(R)
print(w)