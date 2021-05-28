import matplotlib.pyplot as plt
import numpy as np


# ---------- Accessing files ----------

path_to_test = '_results_windmill_testing/'
folder = 'twinlevels4/'
num_trials = 64
sample = 'sample000000'

# contains link to all the sample folders in the test results
folders = [path_to_test + folder + sample + "{:02d}".format(trial) + '/' for trial in range(num_trials)]

# force and velocity values go from 0 to num_windmills - 1
num_windmills = 2

# force files contain
# time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc drag thrust lift perimeter circulation
force_name = 'forceValues_'

# velocity files contain
# t dt CXsim CYsim CXlab CYlab angle u v omega M J accx accy accw
velocity_name = 'velocity_'

# target velocity files contain
# t target_vel
target_name = 'targetvelocity_'

# array of all samples containing array of forces for windmills
# shape num_trials x num_windmills
force_files = [ [folder_ + force_name + str(num) + '.dat' for num in range(num_windmills)] for folder_ in folders]

velocity_files = [ [folder_ + velocity_name + str(num) + '.dat' for num in range(num_windmills)] for folder_ in folders]

target_files = [ [folder_ + target_name + str(num) + '.dat' for num in range(num_windmills)] for folder_ in folders]

# ---------- Loading data ----------

# reminder, data is layed out in columns with time increasing as one goes down

data = np.genfromtxt(velocity_files[0][0], skip_header=1, delimiter=' ')

time = data[:, 0]

num_steps = time.shape[0]

actions = np.zeros((num_steps, num_trials, num_windmills))

states = np.zeros((num_steps, num_trials, num_windmills, 2)) # the 2 here is for angles and angular velocities


for trial in range(num_trials):
  for mill in range(num_windmills):
    forces = np.genfromtxt(force_files[trial][mill], skip_header=1, delimiter=' ')
    velocities = np.genfromtxt(velocity_files[trial][mill], skip_header=1, delimiter=' ')

    actions[:, trial, mill] = forces[:, 7] # tau
    states[:, trial, mill, 0] = velocities[:, 7] # angle
    states[:, trial, mill, 1] = velocities[:, 9] # angvel

  print("Loaded trial ", trial+1)


# ---------- Plots ----------


# action vs time with std and mean (torque vs time)

fig = plt.figure(1)

tau_mean = np.mean(actions, axis=(1, 2))
tau_std = np.std(actions, axis=(1, 2))

plt.plot(time, tau_mean)
plt.fill_between(time, tau_mean - tau_std, tau_mean + tau_std, color='gray', alpha=0.2)
plt.xlabel(r"$t$ [s]")
plt.ylabel(r'$\tau$ []')

# velocity at target point vs time 
dat =np.genfromtxt(target_files[0][0], skip_header=1, delimiter=' ')
t = dat[:, 0]
vel = data[:, 1]

fig2 = plt.figure(2)

plt.plot(t, vel)
plt.xlabel(r"$t$ [s]")
plt.ylabel(r'$v$ []')


# histogram of policy, i.e. action vs state

fig3 = plt.figure(3)

angvel_mean = np.mean(states[:, :, 0, :], axis=1)[:, 1]
angvel_std = np.std(states[:, :, 0, :], axis=1)[:, 1]

plt.plot(angvel_mean, tau_mean)
plt.fill_between(angvel_mean, tau_mean - tau_std, tau_mean + tau_std, color='gray', alpha=0.2)

plt.xlabel(r"$\omega$ [s$^{-1}$]")
plt.ylabel(r'$\tau$ []')

plt.show()
