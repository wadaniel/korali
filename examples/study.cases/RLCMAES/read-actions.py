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
parser.add_argument('--nbins', help='Number of bins', required=True, type=int)
parser.add_argument('--combine', help='Combine histograms', required=False, action='store_true')
parser.add_argument('--marker', help='Red vertical lines.', nargs=3, required=False)

args = parser.parse_args()
 
plt.figure()
fix, (ax1, ax2, ax3) = plt.subplots(3, 1)


bins = np.linspace(0.0,1.0,args.nbins)

cs_tot = np.array([])
cm_tot = np.array([])
cu_tot = np.array([])

for idx, filename in enumerate(args.files):

    print("Read file {}".format(filename))

    if not os.path.isfile(filename):
        print("File {} not found. Abort..".format(filename))
        sys.exit()

    history = np.load(filename)

    actionHistory = history['actionHistory']
    print(actionHistory.shape)
    print("{} Entries found in file {}".format(len(actionHistory), filename))
    
    cs = actionHistory[:,:,0].flatten()
    cs_tot = np.append(cs_tot, cs)
    print(cs.shape)
    cm = actionHistory[:,:,1].flatten()
    cm_tot = np.append(cm_tot, cm)
    print(cs.shape)
    cu = actionHistory[:,:,2].flatten()
    cu_tot = np.append(cu_tot, cu)
 
    if args.combine == False:
        ax1.hist(cs, bins=bins, density=True)
        ax2.hist(cm, bins=bins, density=True)
        ax3.hist(cu, bins=bins, density=True)


if args.combine == True:
    cs_med = np.median(cs_tot)
    cm_med = np.median(cm_tot)
    cu_med = np.median(cu_tot)
    
    marks = [float(m) for m in args.marker]

    ax1.hist(cs_tot, bins=bins, density=True, range=(0,1))
    ax1.axvline(cs_med, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax1.get_ylim()
    ax1.text(cs_med*1.1, max_ylim*0.9, 'Median: {:.2f}'.format(cs_med))
    
    ax2.hist(cm_tot, bins=bins, density=True, range=(0,1))
    ax2.axvline(cm_med, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax2.get_ylim()
    ax2.text(cm_med*1.1, max_ylim*0.9, 'Median: {:.2f}'.format(cm_med))
    
    ax3.hist(cu_tot, bins=bins, density=True, range=(0,1))
    ax3.axvline(cu_med, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax3.get_ylim()
    ax3.text(cu_med*1.1, max_ylim*0.9, 'Median: {:.2f}'.format(cu_med))
 
    if marks:
        ax1.axvline(marks[0], color='r', linestyle='dashed', linewidth=1)
        ax2.axvline(marks[1], color='r', linestyle='dashed', linewidth=1)
        ax3.axvline(marks[2], color='r', linestyle='dashed', linewidth=1)

#ax1.legend()
ax1.title.set_text("Histogram cs")
ax2.title.set_text("Histogram cm")
ax3.title.set_text("Histogram cu")
#ax2.legend()
plt.tight_layout()
plt.savefig(args.out, format='png')

