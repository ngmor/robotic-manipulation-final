#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


filename = 'csv/error.csv'

error_data = np.genfromtxt(filename, delimiter=',').T

#Timestep/bounds
dt = 0.01

times = np.linspace(0.,error_data.shape[1]*dt,error_data.shape[1])

# Get error values
omega_x = error_data[0,:]
omega_y = error_data[1,:]
omega_z = error_data[2,:]
v_x = error_data[3,:]
v_y = error_data[4,:]
v_z = error_data[5,:]

fig = plt.figure('Components of error twist over time')
ax = fig.add_subplot()
ax.set_xlabel(r't (sec)')
ax.set_ylabel(r'Error components')
ax.set_title(r'Components of error twist over time')

ax.plot(times,omega_x,label=r'$\omega_x (rad/s)$')
ax.plot(times,omega_y,label=r'$\omega_y (rad/s)$')
ax.plot(times,omega_z,label=r'$\omega_z (rad/s)$')
ax.plot(times,v_x,label=r'$v_x (m/s)$')
ax.plot(times,v_y,label=r'$v_y (m/s)$')
ax.plot(times,v_z,label=r'$v_z (m/s)$')

ax.legend()
plt.show()