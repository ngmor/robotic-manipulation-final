#!/usr/bin/env python3

import sympy as sym
import numpy as np
import modern_robotics as mr


#2
a3,a4,a5,T = sym.symbols('a3,a4,a5,T')

lhs1 = a3*T**3 + a4*T**4 + a5*T**5
rhs1 = 1
lhs2 = 3*a3*T**2 + 4*a4*T**3 + 5*a5*T**4
rhs2 = 0
lhs3 = 6*a3*T + 12*a4*T**2 + 20*a5*T**3
rhs3 = 0

eqns = sym.Eq(sym.Matrix([lhs1,lhs2,lhs3]),sym.Matrix([rhs1,rhs2,rhs3]))

soln = sym.solve(eqns, [a3,a4,a5],dict=True)

for sol in soln:

    for var in sol:
        print(f'{var} = {sol[var]}')

#5
print(mr.QuinticTimeScaling(5,3))

#6
xstart = np.eye(4,4)
xend = np.array([
    [0,0,1,1],
    [1,0,0,2],
    [0,1,0,3],
    [0,0,0,1],
])

N = 10
Tf = 10
method = 3

traj = mr.ScrewTrajectory(xstart, xend, Tf, N, method)

print(traj[8])

#7
method = 5

traj = mr.CartesianTrajectory(xstart,xend,Tf,N,method)

print(traj[8])