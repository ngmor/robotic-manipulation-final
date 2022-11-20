#!/usr/bin/env python3

import sympy as sym
import numpy as np
import modern_robotics as mr

def hi(r,x,y,beta,gamma,phi):
    return (1 / (r*np.cos(gamma))) * np.array([
        x*np.sin(beta + gamma) - y*np.cos(beta + gamma),
        np.cos(beta + gamma + phi),
        np.sin(beta + gamma + phi),
    ]).T

# Define H(0)
gammai = 0
r = 0.25
beta1 = -np.pi/4
x1 = 2
y1 = 2
h10 = hi(r,x1,y1,beta1,gammai,0)
beta2 = np.pi/4
x2 = -2
y2 = 2
h20 = hi(r,x2,y2,beta2,gammai,0)
beta3 = 3*np.pi/4
x3 = -2
y3 = -2
h30 = hi(r,x3,y3,beta3,gammai,0)
beta4 = -3*np.pi/4
x4 = 2
y4 = -2
h40 = hi(r,x4,y4,beta4,gammai,0)

H0 = np.vstack((h10,h20,h30,h40))
print(H0)

#1
Vb = np.array([1,0,0])
u = H0 @ Vb
print(u)

#2
Vb = np.array([1,2,3])
u = H0 @ Vb
print(u)

#3
vmax = 10 / H0[0][1]
print(vmax)

#5
rmin = 2
vmax = 10

om_max = vmax/rmin
print(om_max)