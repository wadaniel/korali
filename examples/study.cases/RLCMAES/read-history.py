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
parser.add_argument('--out', help='Name of output file.', required=False, type=str)

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

    scaleHistory = np.median(history['scaleHistory'], axis=0)
    muobjectiveHistory = np.median(history['muobjectiveHistory'], axis=0)
    
    objectiveHistoryUpperQuantile = np.quantile(history['objectiveHistory'], 0.8, axis=0)
    objectiveHistoryLowerQuantile = np.quantile(history['objectiveHistory'], 0.2, axis=0)
    objectiveHistory = np.median(history['objectiveHistory'], axis=0)

    gens = np.arange(len(scaleHistory))
 
    col = colors[idx]
    ax.plot(gens, scaleHistory, c=col, linestyle='dotted', linewidth=0.5)
    ax.plot(gens, objectiveHistory, c=col, linestyle='solid', label=filename)
    ax.plot(gens, muobjectiveHistory, c=col, linestyle='dashed', linewidth=0.5)
    ax.fill_between(gens, objectiveHistoryLowerQuantile, objectiveHistoryUpperQuantile, color=col, alpha=0.2)

ax.legend()
output = filename.replace("npz","png")
plt.title("Median of scale (dotted), best objective (solid), and mu-objective (dashed)")
plt.savefig(output, format='png')

