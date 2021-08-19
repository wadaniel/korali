import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parse console arguments
parser = argparse.ArgumentParser()
parser.add_argument('--files', help='History file to read.', required=True, type=str, nargs='+')

args = parser.parse_args()
 
plt.figure()
fix, ax = plt.subplots(1, 1)
ax.set_yscale('log')


colors = ['blue', 'red']

for idx, filename in enumerate(args.files):

    if not os.path.isfile(filename):
        print("File {} not found. Abort..".format(filename))
        sys.exit()

    history = np.load(filename)

    scaleHistory = np.mean(history['scaleHistory'], axis=0)
    objectiveHistory = np.mean(history['objectiveHistory'], axis=0)
    muobjectiveHistory = np.mean(history['muobjectiveHistory'], axis=0)

    gens = np.arange(len(scaleHistory))
    
    ax.plot(gens, scaleHistory, c=colors[idx], linestyle='dotted', linewidth=0.5)
    ax.plot(gens, objectiveHistory, c=colors[idx], linestyle='solid', label=filename)
    ax.plot(gens, muobjectiveHistory, c=colors[idx], linestyle='dashed', linewidth=0.5)

ax.legend()
output = filename.replace("npz","png")
plt.savefig(output, format='png')

