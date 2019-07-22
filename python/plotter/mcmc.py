#! /usr/bin/env python3

import os
import re
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from korali.plotter.helpers import readFiles, verifyRunId, plt_pause_light


# Plot MCMC results (read from .json files)
def plot_mcmc(src, live=False, test=False):
     
    plt.style.use('seaborn-dark')
    init   = False 
    burnin = -1
   
    fig = None
    ax  = None

    resultfiles = readFiles(src)

    if (live == True):
        for filename in resultfiles:
            path   = '{0}/{1}'.format(src, filename)
            
            with open(path) as f:
                data     = json.load(f)
                numdim   = len(data['Variables'])
                burnin   = data['Solver']['Burn In']
                chainlen = data['Solver']['Internal']['Chain Length']
                
                if (init == False):
                    init = True
                    fig, ax = plt.subplots(numdim, numdim, figsize=(8,8))
                    fig.show()
    
                if (not plt.fignum_exists(fig.number)):
                    print("[Korali] Figure closed - Bye!")
                    exit(-1)

                if chainlen > burnin:
                    plot_samples(fig, ax, data, filename)
                    plt_pause_light(0.5) 
            

    if (live == False):
        path = '{0}/{1}'.format(src, resultfiles[-1])
        
        with open(path) as f:
            data     = json.load(f)
            numdim   = len(data['Variables'])
            burnin   = data['Solver']['Burn In']
            chainlen = data['Solver']['Internal']['Chain Length']
            if chainlen <= burnin:
                print("[Korali] Error: No samples found in file {0}...".format(path))
                exit(-1)
            
            fig, ax = plt.subplots(numdim, numdim, figsize=(8,8))
            fig.show()
            plot_samples(fig, ax, data, resultfiles[-1])
   
    if (test == False):
        plt.show() 
    
    print("[Korali] Figure closed - Bye!")


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


# Plot MCMC result file
def plot_samples(fig, ax, data, filename):
    dims     = data['Variables']
    numdim   = len(dims)
    pop      = data['Solver']['Internal']['Database Entry Count']
    samples  = np.reshape( data['Solver']['Internal']['Sample Parameters Database'][0:pop*numdim], (pop,numdim) )
    
    fig.canvas.set_window_title(filename)
    
    plt.suptitle( 'MCMC\nNumber of Samples {0}\n'.format(str(pop)),
                  fontweight='bold',
                  fontsize  = 12)

    plot_histogram(ax, samples)
    plot_upper_triangle(ax, samples, False)
    plot_lower_triangle(ax, samples)
