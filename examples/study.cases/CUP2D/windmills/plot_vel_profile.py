import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

true_prof = np.array([0.0117094, 0.0118713, 0.0120126, 0.0121298, 0.0122235, 0.0122405, 0.0116373, 0.00707957,
                                       0.0125024, 0.0537881, 0.0965903, 0.101789, 0.0676126, 0.0271084, 0.00552582, 0.00977532,
                                       0.0129954, 0.0133042, 0.0114955, 0.00804636, 0.00881718, 0.0221757, 0.0595386, 0.107892,
                                       0.102329, 0.0453045, 0.0106931, 0.0173096, 0.0200404, 0.0201389, 0.0199034, 0.0197252])
""""""
pwd_orig = "/scratch/snx3000/anoca/CUP2D/4hzsym/"
pwd = "/scratch/snx3000/anoca/korali/vel_oui_mieux/_testingResults/sample00000000/"
file_name = "velocity_profile_0.dat"
data_orig = np.genfromtxt(pwd_orig + file_name, delimiter=' ')

print(data_orig.shape)

f = plt.figure()

# low is at 0.35, high is at 1.05
# 0.7/32 = 0.021875 = height of each interval
height = 0.021875
x = np.linspace(0.35 + height/2, 1.05 - height/2, 32)

profile_orig = plt.plot(x, data_orig[0, 1:], label='orig_profile')[0]
avg = np.mean(data_orig[:, 1:], axis=0)
plt.plot(x, avg, label='avg')
# profile = plt.plot(x, data[0, 1:], label='profile')[0]
# target = plt.plot(x, data_orig[-1, 1:], label='target')
plt.xlabel('xpos')
plt.ylabel('|v|')
plt.ylim(-0.01,0.45)
plt.show()
plt.legend()

#plt.savefig('png/freqsquare2hz.png')

t = data_orig[:,0]

# delta_t = t[1:] - t[:-1]
# v_delta_t = data_orig[1:, 1:] * delta_t
# sum_v = np.cumsum(v_delta_t, axis=0)

# T = np.cumsum(t)

# time_average = sum_v / T

f2 = plt.figure()


plt.plot(t, data_orig[:, 16])
plt.xlabel('t')
plt.ylabel('|v|')
plt.ylim(-0.01,0.45)
plt.savefig('png/4hzsym.png')



# function takes frame as an input
def AnimationFunction(frame):
    # line is set with new values of x and y
    profile_orig.set_data((x, data_orig[frame, 1:]))
    #profile_orig.set_data((x, time_average))

anim_created = FuncAnimation(f, AnimationFunction, frames=199, interval=100, save_count=199)

anim_created.save('gif/4hzsym.gif')

""""""

# compute the reward obtained for this particular case

"""pwd = "/scratch/snx3000/anoca/CUP2D/forced_rot_long40/"
file_name = "velocity_profile_0.dat"
data = np.genfromtxt(pwd + file_name, delimiter=' ')

profiles = data[:, 1:]
diff = true_prof - profiles
square_diff = diff * diff
D = 10
sum_square_diff = -D * np.sum(np.sum(square_diff))
print(sum_square_diff)"""