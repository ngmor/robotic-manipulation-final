#!/usr/bin/env python3

import sympy as sym
import numpy as np
import modern_robotics as mr


#1
M = 1
b = 2
Kd = 3
Kp = 4

Ki = (b + Kd)*Kp / M
print(Ki)

#3
s = sym.symbols('s')
print(((s + 4)**3).expand())

Kd = 12*M - b
print(Kd)