#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_file: str,
         target_radius: float):
    df = pd.read_csv(csv_file)

    nabfs = len(df.columns) // 2

    fig, ax = plt.subplots()

    for i in range(nabfs):
        x = df[f"x{i}"].to_numpy()
        y = df[f"y{i}"].to_numpy()
        ax.plot(x, y)

    target = plt.Circle((0, 0), target_radius, color='r', alpha = 0.3)
    ax.add_artist(target)

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
