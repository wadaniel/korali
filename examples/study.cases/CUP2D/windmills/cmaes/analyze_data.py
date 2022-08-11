import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

from helpers import average_profile, mse

name = 'slowdiff/'
folder='/scratch/snx3000/anoca/CUP2D/' + name
files = ['x_velocity_profile_0.dat', 'y_velocity_profile_0.dat']
output_folder = 'results/' + name

x_vel = np.genfromtxt(folder + files[0], delimiter=' ', skip_header=1)

plt.figure(1)
plt.plot(x_vel[:, 0], x_vel[:, 5])
plt.savefig(output_folder + 'random_profile_x.png')

plt.figure(2)
res_x = average_profile(x_vel, 720, 1200)
plt.plot(res_x)
plt.savefig(output_folder + 'averaged_profile_x.png')

np.savetxt( output_folder + 'x_profile.dat', res_x[[-1], :])

y_vel = np.genfromtxt(folder + files[1], delimiter=' ', skip_header=1)

plt.figure(3)
plt.plot(y_vel[:, 0], y_vel[:, 5])
plt.savefig(output_folder + 'random_profile_y.png')

plt.figure(4)
res_y = average_profile(y_vel, 720, 1200)
plt.plot(res_y)
plt.savefig(output_folder + 'averaged_profile_y.png')

dat = np.genfromtxt(folder + 'velocity_0.dat', delimiter=' ', skip_header=1)
omega = dat[:, 9]

np.savetxt(output_folder + 'y_profile.dat', res_y[[-1], :])

plt.figure(5)
plt.plot(omega)
plt.savefig(output_folder + 'omega.png')






# subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
# subfolders.remove('/scratch/snx3000/anoca/CUP2D/old')
# subfolders.sort()

# fans2_folders = subfolders[:-3]
# fans4_folders = subfolders[-3:]

files = [ f.path for f in os.scandir(folder) ]

# folders = fans4_folders
# folders = [folder + "2fans_m1"]
# output_folder = '2fans/'

##### Plot velocity_profile of 2 fans 

# fans2_folders = fans2_folders[0:1]
# print(fans2_folders)

# for folder in folders:
#     data = np.genfromtxt(folder + "/velocity_profile_0.dat", delimiter=' ', skip_header=1)

#     t = data[:, 0]
#     profile = data[:, 1:]

#     fig_profile = plt.figure()

#     extent = [0, 31, t[-1], 0]
#     plt.imshow(profile, aspect='auto', extent=extent)
#     plt.xlabel('pos')
#     plt.ylabel('time')
#     plt.colorbar()

#     folder_name = os.path.basename(folder)
#     plt.savefig(output_folder + 'profile_' + folder_name + '.png')

##### Plot the average velocity_profile of 2 fans

# write in velocity profile folder


# folder_saveprof = 'avgprofiles/'
# avgprofiles = np.zeros((11, 32))

# fig, axs = plt.subplots(2, 6, figsize=(15,6))
# first_row = ['2fans_m1.dat', '2fans_m08.dat', '2fans_m06.dat', '2fans_m04.dat', '2fans_m02.dat', '2fans_0.dat']
# second_row = ['2fans_02.dat', '2fans_04.dat', '2fans_06.dat', '2fans_08.dat', '2fans_1.dat']

# for index, file in enumerate(first_row):
#     data = np.genfromtxt(folder + file, delimiter=' ', skip_header=1)

#     rows = 600 * 20 # 600s of simulation

#     avgprofile = average_profile(data, 0, rows)

#     # store the average profile at time 200s (100th period) in the list of average profiles
#     avgprofiles[index, :] = avgprofile[100, :]

#     t = data[:, 0]
#     extent = [0, 31, t[rows], 0]
#     im = axs[0, index].imshow(avgprofile, aspect='auto', extent=extent, cmap='plasma', vmin=0, vmax=0.28)
#     axs[0, index].set(xlabel='pos')
#     if index != 0:
#         axs[0, index].axes.yaxis.set_visible(False)
#     else:
#         axs[0, index].set(ylabel='time')

# for index, file in enumerate(second_row):
#     data = np.genfromtxt(folder + file, delimiter=' ', skip_header=1)

#     rows = 600 * 20 # 600s of simulation

#     avgprofile = average_profile(data, 0, rows)

#     # store the average profile at time 200s (100th period) in the list of average profiles
#     avgprofiles[index + 6, :] = avgprofile[100, :]

#     t = data[:, 0]
#     extent = [0, 31, t[rows], 0]
#     im = axs[1, index].imshow(avgprofile, aspect='auto', extent=extent, cmap='plasma', vmin=0, vmax=0.28)
#     axs[1, index].set(xlabel='pos')
#     if index != 0:
#         axs[1, index].axes.yaxis.set_visible(False)
#     else:
#         axs[1, index].set(ylabel='time')
    
#     if index == 4:
#         fig.colorbar(im, ax=axs[1, index+1])
#         fig.delaxes(axs[1, index+1])

# plt.savefig(output_folder + 'avgprofile_all.png')

# np.savetxt(folder_saveprof + 'avgprofiles.dat', avgprofiles)



# for file in files:
#     data = np.genfromtxt(file, delimiter=' ', skip_header=1)

#     rows = data.shape[0]

#     avgprofile = average_profile(data, 0, rows)

#     t = data[:, 0]

#     fig_avgprofile = plt.figure()

#     extent = [0, 31, t[-1], 0]
#     plt.imshow(avgprofile, aspect='auto', extent=extent)
#     plt.xlabel('pos')
#     plt.ylabel('time')
#     plt.colorbar()
#     plt.clim(0,0.28)

#     file_name = os.path.splitext(os.path.basename(file))[0]
#     plt.savefig(output_folder + 'avgprofile_' + file_name + '.png')

##### Plot the mse


# fig, axs = plt.subplots(2, 6, figsize=(15,6))
# first_row = ['2fans_m1.dat', '2fans_m08.dat', '2fans_m06.dat', '2fans_m04.dat', '2fans_m02.dat', '2fans_0.dat']
# second_row = ['2fans_02.dat', '2fans_04.dat', '2fans_06.dat', '2fans_08.dat', '2fans_1.dat']

# for index, file in enumerate(first_row):
#     data = np.genfromtxt(folder + file, delimiter=' ', skip_header=1)

#     rows = 600 * 20 # 600s of simulation

#     mse_ = mse(data, 0, rows)
#     axs[0, index].plot(mse_)

#     axs[0, index].set(xlabel='periods')
#     if index == 0:
#         axs[0, index].set(ylabel='mse')

# for index, file in enumerate(second_row):
#     data = np.genfromtxt(folder + file, delimiter=' ', skip_header=1)

#     rows = 600 * 20 # 600s of simulation

#     mse_ = mse(data, 0, rows)
#     axs[1, index].plot(mse_)

#     axs[1, index].set(xlabel='periods')
#     if index == 0:
#         axs[1, index].set(ylabel='mse')

#     if index == 4:
#         fig.delaxes(axs[1, index+1])

# plt.savefig(output_folder + 'mse_all.png')

# for folder in folders:
#     print(folder)
#     data = np.genfromtxt(folder + "/velocity_profile_0.dat", delimiter=' ', skip_header=1)

#     rows = data.shape[0]

#     mse_ = mse(data, 0, rows)

#     fig_mse = plt.figure()

#     plt.plot(mse_)
#     plt.xlabel('periods')
#     plt.ylabel('mse')

#     folder_name = os.path.basename(folder)
#     plt.savefig(output_folder + 'mse_' + folder_name + '.png')


##### Plot the angular velocity of 2 fans

# for folder in folders:
#     data = np.genfromtxt(folder + "/velocity_0.dat", delimiter=' ', skip_header=1)

#     t = data[:, 0]
#     angvel = data[:, 9]

#     fig_profile = plt.figure()

#     plt.plot(t, angvel)
#     plt.xlabel('time')
#     plt.ylabel('Angular velocity')

#     folder_name = os.path.basename(folder)
#     plt.savefig(output_folder + 'angvel_' + folder_name + '.png')



##### Plot the avg profiles at time 200s in subplots

# folder_saveprof = 'avgprofiles/avgprofiles.dat'

# avgprofiles_200 = np.loadtxt(folder_saveprof)

# rows = avgprofiles_200.shape[0]

# fig, axs = plt.subplots(2, 6, figsize=(15,6))

# for index, file in enumerate(range(rows)):

#     axs[index // 6 , index % 6].plot(avgprofiles_200[index, :])
#     axs[index // 6 , index % 6].set(xlabel='pos')
#     axs[index // 6 , index % 6].set_ylim([0, 0.25])
#     if index % 6 != 0:
#         axs[index // 6 , index % 6].axes.yaxis.set_visible(False)
#     else:
#         axs[index // 6 , index % 6].set(ylabel='|v|')

#     if index == 10:
#         fig.delaxes(axs[1, 5])


# plt.savefig(output_folder + 'avgprofile_200.png')


# ##### Plot the avg profiles at time 200s in one colorplot

# folder_saveprof = 'avgprofiles/avgprofiles.dat'

# avgprofiles_200 = np.loadtxt(folder_saveprof)

# rows = avgprofiles_200.shape[0]

# x = np.linspace(0.3, 1.5, rows)

# fig = plt.figure()

# extent = [0, 31, 1.1, -1.1]
# plt.imshow(avgprofiles_200, aspect='auto', extent=extent, cmap='plasma')
# plt.xlabel('pos')
# plt.ylabel('factor')
# plt.colorbar()
# ax = plt.gca()
# ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])


# # for i in range(rows):
# #     plt.plot(avgprofiles_200[i, :], color=lighten_color("blue", x[i]))

# plt.savefig(output_folder + 'avgprofile_200_colormap.png')



##### Compute and plot the normalized mses (of the average) over time

