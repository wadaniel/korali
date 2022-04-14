import numpy as np
import matplotlib.pyplot as plt

folder = "profiles/"
file_ = ["profile_" + str(i) + ".dat" for i in range(11)]

for ind, file in enumerate(file_):
    data = np.genfromtxt(folder + file, delimiter=' ')

    # divide the cumulative sum of the values by the time => time average essentially
    t = data[:4000, 0:1]
    cumulative_summed = np.cumsum(data[:4000, 1:], axis = 0) / t; 
    np.savetxt("avgprofiles/avg" + file, cumulative_summed)

    # if(ind == 0):
    #     plt.figure()
    #     plt.plot(t, cumulative_summed[:, 12])
    #     plt.savefig("test.png")


