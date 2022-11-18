#!/usr/bin/env python3

import sympy as sym
import numpy as np
import modern_robotics as mr

#2
A = np.array([
    [0,1,0],
    [0,0,1],
    [-3,-2,-1]
])

s = sym.symbols('s')
I = np.eye(3,3)
eq = (sym.Matrix(A) - s*sym.Matrix(I)).det()*-1
print(eq)
roots = np.linalg.eigvals(A)
for root in roots:
    print(root)

#5
a = 2
b = 2
wn = np.sqrt(b)
c = a / (2 * wn)
print(f'Natural frequency: {wn}')
print(f'Damping ratio: {c}')

#6
a = 3
b = 9
wn = np.sqrt(b)
c = a / (2 * wn)
wd = wn * np.sqrt(1 - c**2)
print(f'Damped natural frequency: {wd}')

#8
Ki = 10
Kp = 2*np.sqrt(Ki)
print(f'Kp = {Kp}')

#10
c = Kp / (2*np.sqrt(Ki))
wn = np.sqrt(Ki)
ts = 4.0 / (c*wn)
print(f'2% Settling time: {ts}')

#11
Kp = 20
Ki = (Kp / 2.0)**2
c = Kp / (2*np.sqrt(Ki))
wn = np.sqrt(Ki)
ts = 4.0 / (c*wn)
print(f'2% Settling time: {ts}')
