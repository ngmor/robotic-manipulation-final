#!/usr/bin/env python3

import sympy as sym
import numpy as np
import modern_robotics as mr

#1
x,y = sym.symbols('x,y')

f = sym.Matrix([x**2 - 9, y**2 - 4])

Jinv = f.jacobian([x,y]).inv()

f_lam = sym.lambdify([x,y],f)
Jinv_lam = sym.lambdify([x,y],Jinv)

theta0 = np.array([[1,1]]).T

theta1 = theta0 - np.matmul(Jinv_lam(theta0[0][0],theta0[1][0]), f_lam(theta0[0][0],theta0[1][0]))
#print(theta1)
theta2 = theta1 - np.matmul(Jinv_lam(theta1[0][0],theta1[1][0]), f_lam(theta1[0][0],theta1[1][0]))
print('1:')
print(theta2)


#2
eps_w = 0.001
eps_v = 0.0001
theta0 = np.array([np.pi/4, np.pi/4, np.pi/4])
M = np.array([[1,0,0,3],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]])
Tsd = np.array([[-0.585,-0.811,0,0.076],
                [0.811,-0.585,0,2.608],
                [0,0,1,0],
                [0,0,0,1]])

Blist = np.array([[0,0,1,0,3,0],
                  [0,0,1,0,2,0],
                  [0,0,1,0,1,0]]).T

[thetalist, success] = mr.IKinBody(M=M,Blist=Blist,T=Tsd,thetalist0=theta0,eomg=eps_w,ev=eps_v)

print("2:")
print(thetalist)