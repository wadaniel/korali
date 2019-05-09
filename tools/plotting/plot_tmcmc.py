#! /usr/bin/env python3

import os
import re
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from plot_helpers import plt_pause_light

# Plot histogram of sampes in diagonal
def plot_histogram(ax, theta):
    dim = theta.shape[1]
    num_bins = 50
    
    for i in range(dim):

        if (dim == 1): 
            ax_loc = ax
        else: 
            ax_loc = ax[i,i]
 
        hist, bins, _ = ax_loc.hist(theta[:, i], num_bins, density=True,
                                    color='lightgreen', ec='black')
        
        if i == 0:
           
            # Rescale hist to scale of theta -> get correct axis titles
            widths = np.diff(bins)
            if (dim > 1) :  
                hist = hist / np.max(hist) * (ax_loc.get_xlim()[1] -
                                                ax_loc.get_xlim()[0])
                bottom = ax_loc.get_xlim()[0]
                ax_loc.cla()
                ax_loc.bar(bins[:-1], hist, widths,
                    color='lightgreen', ec='black', bottom=bottom)
                ax_loc.set_ylim(ax_loc.get_xlim())
                ax_loc.set_xticklabels([])
            else:
                ax_loc.cla()
                ax_loc.bar(bins[:-1], hist, widths,
                     color='lightgreen', ec='black')

        elif i == theta.shape[1] - 1:
            ax_loc.set_yticklabels([])
        
        else:
            ax_loc.set_xticklabels([])
            ax_loc.set_yticklabels([])
        ax_loc.tick_params(axis='both', which='both', length=0)


#Plot scatter plot in upper triangle of figure
def plot_upper_triangle(ax, theta, lik=False):
    dim = theta.shape[1]
    if (dim == 1): return
    
    for i in range(dim):
        for j in range(i + 1, dim):
            if lik:
                ax[i, j].scatter(theta[:, j], theta[:, i], marker='o', s=10,
                    c=theta, alpha=0.5)
            else:
                ax[i, j].plot(theta[:, j], theta[:, i], '.', markersize=1)
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])
            ax[i, j].grid(b=True, which='both')


#Plot 2d histogram in lower triangle of figure
def plot_lower_triangle(ax, theta):
    dim = theta.shape[1]
    if (dim == 1): return

    for i in range(dim):
        for j in range(i):
            # returns bin values, bin edges and bin edges
            H, xe, ye = np.histogram2d(theta[:, j], theta[:, i], 10, density=True)
            # plot and interpolate data
            ax[i, j].imshow(H.T, aspect="auto", interpolation='spline16',
                            origin='lower', extent=np.hstack((
                                                ax[j, j].get_xlim(),
                                                ax[i, i].get_xlim())),
                                                cmap=plt.get_cmap('jet'))
            
            if i < theta.shape[1]-1:
                ax[i, j].set_xticklabels([])
            if j > 0:
                ax[i, j].set_yticklabels([])


# Plot TMCMC result file
def plot_samples(path, idx=None):
    with open(path) as f:
        data    = json.load(f)
        numdim  = data['Dimension']
        pop     = data['Solver']['Population Size']
        state   = data['Solver']['State']
        anneal  = state['AnnealingExponent']
        fitness = state['DatabaseFitness']
        samples = np.reshape( state['DatabasePoints'], (pop,numdim) )

        plt.style.use('seaborn-dark')
        fig, ax = plt.subplots(samples.shape[1], samples.shape[1], figsize=(8,8))
        fig.show()
        
        if idx is None: 
            fig.canvas.set_window_title(path)
            idx = int(re.findall(r'[0-9]+', path)[-1])
        else:           
            fig.canvas.set_window_title('s{0}.json'.format(str(idx).zfill(5)))
        
        plt.suptitle( 'Samples Generation {0}\n (Annealing Exponent {1:.4e})'.format(str(idx),\
                            anneal),
                      fontweight='bold',
                      fontsize  = 12)

        plot_histogram(ax, samples)
        plot_upper_triangle(ax, samples, False)
        plot_lower_triangle(ax, samples)
        return anneal

# Plot TMCMC results (read from .json files)
def run_diagnostics(src, live=False):
     
    idx      = 0
    finished = False
    
    while( not finished ):

        path = '{0}/s{1}.json'.format(src, str(idx).zfill(5))
        if ( not os.path.isfile(path) ):
            if ( live == True ):
                if (idx > 1): plt_pause_light(0.05)
                continue
            else: 
                break

        time.sleep(0.1)
        anneal = plot_samples(path, idx)
        
        plt_pause_light(0.05) 
        if(live == False): time.sleep(0.1)
        idx = idx+1
        if (anneal >= 1.0): finished = True

    plt.show()


if __name__ == '__main__':

    if (len(sys.argv) == 2):
        if os.path.isdir(sys.argv[1]):
            run_diagnostics(src=sys.argv[1], live=True)
        
        elif os.path.isfile(sys.argv[1]):
            print  ("Plotting results from file " + sys.argv[1])
            plot_samples(sys.argv[1])
            plt.show()

        else:
            print("Invalid arguments, exit ...")
            print("argument not recognized as directory or file")

    else: 
        print("Invalid arguments, exit ...")
        print("one argument required, directoy of TMCMC results or file")
