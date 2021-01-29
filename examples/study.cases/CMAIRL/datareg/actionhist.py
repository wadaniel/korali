#!/usr/bin/env python3

import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readActions(obsfile):

 with open(obsfile) as csv_file:
    obsactions = []
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        obsactions.append(row[4])

    return obsactions


def plotActions(actions, filename):

    plt.hist(actions, cumulative=True, density=True, bins=50, alpha=0.3)  # arguments are passed to np.histogram
    plt.title("Cartpole action histogram")
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', help='Observation file to read', required=True)

    args = parser.parse_args()

    files = args.files
    for idx, fname in enumerate(files):
        actions = readActions(fname)
        filename = "actions_{}.png".format(idx)
        plotActions(actions, filename)
