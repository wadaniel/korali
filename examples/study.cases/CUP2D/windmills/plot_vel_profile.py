import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

true_prof = np.array([0.0117094, 0.0118713, 0.0120126, 0.0121298, 0.0122235, 0.0122405, 0.0116373, 0.00707957,
                                       0.0125024, 0.0537881, 0.0965903, 0.101789, 0.0676126, 0.0271084, 0.00552582, 0.00977532,
                                       0.0129954, 0.0133042, 0.0114955, 0.00804636, 0.00881718, 0.0221757, 0.0595386, 0.107892,
                                       0.102329, 0.0453045, 0.0106931, 0.0173096, 0.0200404, 0.0201389, 0.0199034, 0.0197252])

#pwd = "/scratch/snx3000/anoca/CUP2D/forced_rot_4hz/"
pwd = "/scratch/snx3000/anoca/korali/test5e4/_testingResults/sample00000000/"
file_name = "velocity_profile_0.dat"
data = np.genfromtxt(pwd + file_name, delimiter=' ')[::10, :]
print(data.shape)
# diff = -sum(sum(np.abs(true_prof - data[:, 1:])))
# print(diff)

f = plt.figure()

# low is at 0.35, high is at 1.05
# 0.7/32 = 0.021875 = height of each interval
height = 0.021875
x = np.linspace(0.35 + height/2, 1.05 - height/2, 32)

profile = plt.plot(x, data[0, 1:])[0]
plt.xlabel('xpos')
plt.ylabel('|v|')
plt.ylim(-0.01,0.25)
plt.show()

# plt.savefig('profile.png')



# function takes frame as an input
def AnimationFunction(frame):
    # line is set with new values of x and y
    profile.set_data((x, data[frame, 1:]))

anim_created = FuncAnimation(f, AnimationFunction, frames=220, interval=100, save_count=220)

anim_created.save('profile_5e4.gif')