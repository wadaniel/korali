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
parser.add_argument('--out', help='Name of output file.', required=True, type=str)

args = parser.parse_args()
 
plt.figure()
fix, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_yscale('log')


colors = ['blue', 'red']

for idx, filename in enumerate(args.files):

    print("Read file {}".format(filename))

    if not os.path.isfile(filename):
        print("File {} not found. Abort..".format(filename))
        sys.exit()

    history = np.load(filename)

    print("{} Entries found in file {}".format(len(history['scaleHistory'][0]), filename))
    scaleHistory = np.median(history['scaleHistory'], axis=0)
    meanobjectiveHistory = np.median(history['meanobjectiveHistory'], axis=0)
    actionHistoryMedian = np.median(history['actionHistory'], axis=0)
    actionHistoryUpperQuantile = np.quantile(history['actionHistory'], 0.8, axis=0)
    actionHistoryLowerQuantile = np.quantile(history['actionHistory'], 0.2, axis=0)
    
    objectiveHistoryUpperQuantile = np.quantile(history['objectiveHistory'], 0.8, axis=0)
    objectiveHistoryLowerQuantile = np.quantile(history['objectiveHistory'], 0.2, axis=0)
    objectiveHistory = np.median(history['objectiveHistory'], axis=0)

    gens = np.arange(len(scaleHistory))

    col = colors[idx]

    # objective plot
    ax1.plot(gens, scaleHistory, c=col, linestyle='dotted', linewidth=0.5)
    ax1.plot(gens, objectiveHistory, c=col, linestyle='solid', label=filename)
    ax1.plot(gens, meanobjectiveHistory, c=col, linestyle='dashed', linewidth=0.5)
    ax1.fill_between(gens, objectiveHistoryLowerQuantile, objectiveHistoryUpperQuantile, color=col, alpha=0.2)

    # action plot
    ax2.plot(gens, actionHistoryMedian[:,0], c=col, linestyle='dotted', linewidth=0.5, label="cs")
    ax2.fill_between(gens, actionHistoryLowerQuantile[:,0], actionHistoryUpperQuantile[:,0], color=col, alpha=0.2)
    
    ax2.plot(gens, actionHistoryMedian[:,1], c=col, linestyle='solid', linewidth=0.5, label="cm")
    ax2.fill_between(gens, actionHistoryLowerQuantile[:,1], actionHistoryUpperQuantile[:,1], color=col, alpha=0.2)
    
    ax2.plot(gens, actionHistoryMedian[:,2], c=col, linestyle='dashed', linewidth=0.5, label="cu")
    ax2.fill_between(gens, actionHistoryLowerQuantile[:,2], actionHistoryUpperQuantile[:,2], color=col, alpha=0.2)

ax1.legend()
ax1.title.set_text("Median of scale (dotted), best eval (solid), and mean-eval (dashed)")
ax2.legend()
plt.tight_layout()
plt.savefig(args.out, format='png')

