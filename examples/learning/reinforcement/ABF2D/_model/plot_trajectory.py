#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_file: str,
         target_radius: float,
         start_radius: float=1,
         minspace: float=3):
    df = pd.read_csv(csv_file)

    nabfs = len(df.columns) // 2

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    d = minspace
    xmin = -d
    xmax = d
    ymin = -d
    ymax = d

    for i in range(nabfs):
        x = df[f"x{i}"].to_numpy()
        y = df[f"y{i}"].to_numpy()
        ax.plot(x, y)

        start = plt.Circle((x[0], y[0]), start_radius, color='g', alpha = 0.3)
        ax.add_artist(start)

        xmin = min([xmin, min(x)-d])
        xmax = max([xmax, max(x)+d])
        ymin = min([ymin, min(y)-d])
        ymax = max([ymax, max(y)+d])

    target = plt.Circle((0, 0), target_radius, color='r', alpha = 0.3)
    ax.add_artist(target)

    _min = min([xmin, ymin])
    _max = max([xmax, ymax])

    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('traj', type=str, help='The trajectory file, in csv format.')
    parser.add_argument('--target-radius', type=float, default=1)
    args = parser.parse_args()

    plot(args.traj, args.target_radius)
