import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

from helpers import average_profile, mse

name = 'quickdiff/'
folder='/scratch/snx3000/anoca/CUP2D/' + name
folder_testing='/scratch/snx3000/anoca/CUP2D/testing_' + name
files = ['x_velocity_profile_0.dat', 'y_velocity_profile_0.dat']
output_folder = 'results/' + name


# compare the time averaged profiles at the end
# between target and test

x_vel = np.genfromtxt(folder + files[0], delimiter=' ', skip_header=1)
x_vel_testing = np.genfromtxt(folder_testing + files[0], delimiter=' ', skip_header=1)

plt.figure(1)
res_x = average_profile(x_vel, 700, 1200)
res_x_testing = average_profile(x_vel_testing, 700, 1200)
plt.plot(res_x[-1, :])
plt.plot(res_x_testing[-1, :])
plt.xlabel('profile position')
plt.ylabel('x-velocity')
plt.legend(['target', 'test'])

plt.savefig(output_folder + 'comp_x.png')

y_vel = np.genfromtxt(folder + files[1], delimiter=' ', skip_header=1)
y_vel_testing = np.genfromtxt(folder_testing + files[1], delimiter=' ', skip_header=1)

plt.figure(2)
res_y = average_profile(y_vel, 700, 1200)
res_y_testing = average_profile(y_vel_testing, 700, 1200)
plt.plot(res_y[-1, :])
plt.plot(res_y_testing[-1, :])
plt.xlabel('profile position')
plt.ylabel('y-velocity')
plt.legend(['target', 'test'])


plt.savefig(output_folder + 'comp_y.png')