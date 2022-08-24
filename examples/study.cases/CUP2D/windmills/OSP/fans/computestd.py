import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Ellipse


# load the data, compute the standard deviation, and save it to files
data = np.load('ordered_data.npy') # 41 x 21 x 61 x 4 x 32 x 12

std_per_step = np.std(data, axis=(0, 1))
std_all_steps = np.std(data, axis = (0, 1, 2))

np.save('std_per_step.npy', std_per_step)
np.save('std_all_steps.npy', std_all_steps)


# plot the standard deviation of the data
def createDomain(ax):
    ax.set_xlim([0, 0.525])
    ax.set_ylim([0, 0.7])
    smajax = 0.0405
    sminax = 0.0135
    frac = 0.55

    d = smajax * (1. -(2./3.) * frac)
    num_fans = 2
    centers = [[0.1, 0.25], [0.1, 0.45]]

    # profile = Rectangle((0.35, 0.175), 0.0875, 0.35)
    # profile.set_facecolor('white')
    # profile.set_edgecolor('black')

    # ax.add_patch(profile)

    for i in range(num_fans):
        # bottom ellipse
        center1 = [d * np.sin(np.pi/6), -d * np.cos(np.pi/6)]
        bot_ellipse = Ellipse((centers[i][0] + center1[0], centers[i][1] + center1[1]), 2*smajax, 2*sminax, angle=-60)
        bot_ellipse.set_color('gray')
        ax.add_patch(bot_ellipse)

        # top ellipse
        center2 = [d * np.sin(np.pi/6), d * np.cos(np.pi/6)]
        top_ellipse = Ellipse((centers[i][0] + center2[0], centers[i][1] + center2[1]), 2*smajax, 2*sminax, angle=60)
        top_ellipse.set_color('gray')
        ax.add_patch(top_ellipse)

        # top ellipse
        center3 = [-d, 0]
        middle_ellipse = Ellipse((centers[i][0] + center3[0], centers[i][1] + center3[1]), 2*smajax, 2*sminax, angle=0)
        middle_ellipse.set_color('gray')
        ax.add_patch(middle_ellipse)

    ax.set_aspect('equal')


# results for all the steps

vorticity = std_all_steps[0, :, :]
pressure = std_all_steps[1, :, :]
velx = std_all_steps[2, :, :]
vely = std_all_steps[3, :, :]

output_folder = 'plots/'

# vorticities
fig, ax = plt.subplots()

createDomain(ax)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), vorticity)
plt.colorbar()
plt.title('Standard deviation : vorticity')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'std_vorticity.png')

# pressure
fig2, ax2 = plt.subplots()

createDomain(ax2)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), pressure)
plt.colorbar()
plt.title('Standard deviation : pressure')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'std_pressure.png')

# pressure
fig3, ax3 = plt.subplots()

createDomain(ax3)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), velx)
plt.colorbar()
plt.title('Standard deviation : velx')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'std_velx.png')

# pressure
fig4, ax4 = plt.subplots()

createDomain(ax4)

plt.contourf(np.linspace(0.2625, 0.525, 12), np.linspace(0, 0.7, 32), vely)
plt.colorbar()
plt.title('Standard deviation : vely')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig(output_folder + 'std_vely.png')