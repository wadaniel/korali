#! /usr/bin/env python3

import os
import sys
import glob
import time
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from korali.plotter.helpers import readFiles, hlsColors, pauseLight, drawMulticoloredLine


# Plot CMAES results (read from .json files)
def plot_cmaes(src, plot_mean = False, live=False, test=False ):

    init     = False # init flag
    numdim   = 0  # problem dimension
    names    = [] # description params
    colors   = [] # rgb colors
    numeval  = [] # number obj function evaluations
    sigma    = [] # scaling parameter
    cond     = [] # condition of C (largest EW / smallest EW)
    psL2     = [] # conjugate evolution path L2 norm
    dfval    = [] # abs diff currentBest - bestEver
    fval     = [] # best fval current generation
    fvalXvec = [] # location fval
    mu       = [] # current mean
    axis     = [] # sqrt(EVals)
    ssdev    = [] # sigma x diag(C)
 
    plt.style.use('seaborn-dark')
 
    fig = None
    ax  = None
 
    resultfiles = readFiles(src)

    for filename in resultfiles:
        path   = '{0}/{1}'.format(src, filename)
        
        with open(path) as f:

            data       = json.load(f)
            solverName = data['Solver']['Type']

            state = data['Solver']['Internal']
            gen   = data['General']['Current Generation']

            if (init == False):
    
                init   = True
                numdim = len(data['Variables'])
                names  = [ data['Variables'][i]['Name'] for i in range(numdim) ]
                colors = hlsColors(numdim)

                for i in range(numdim):
                    fvalXvec.append([])
                    mu.append([])
                    axis.append([])
                    ssdev.append([])

                if (live == True):
                    fig, ax = plt.subplots(2,2,num='{0} live diagnostics'.format(solverName), figsize=(8,8))
                    fig.show()

            if ( (live == True) and (not plt.fignum_exists(fig.number))):
                print("[Korali] Figure closed - Bye!")
                exit(0)
    
            if gen > 1:

                numeval.append(data['General']['Function Evaluation Count'])
                dfval.append(abs(state['Current Best Value'] - state['Best Ever Value']))
                
                fval.append(state['Current Best Value'])
                sigma.append(state['Sigma'])
                cond.append(state['Maximum Covariance Eigenvalue']/state['Minimum Covariance Eigenvalue'])
                psL2.append(state['Conjugate Evolution Path L2 Norm'])

                for i in range(numdim):
                    fvalXvec[i].append(state['Current Best Coordinates'][i])
                    mu[i].append(state['Current Mean'][i])
                    axis[i].append(state['Axis Lengths'][i])
                    ssdev[i].append(sigma[-1]*np.sqrt(state['Covariance Matrix'][i*numdim+i]))
            
                if (live == True and gen > 1):
                    draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, mu, axis, ssdev, colors, names, plot_mean, live)
                    pauseLight(0.05)

    if (live == False):
        fig, ax = plt.subplots(2,2,num='{0} live diagnostics'.format(solverName), figsize=(8,8))
        draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, mu, axis, ssdev, colors, names, plot_mean, live)
        fig.show()
    
    plt.show()
    print("[Korali] Figure closed - Bye!")


# Create Plot from Data
def draw_figure(fig, ax, src, idx, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, mu, axis, ssdev, colors, names, plot_mean, live):

    plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)),\
                      fontweight='bold',\
                      fontsize=12 )

    # Upper Left Plot
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')
    drawMulticoloredLine(ax[0,0], numeval, fval, 0.0, 'r', 'b', '$| F |$')
    ax[0,0].plot(numeval, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
    ax[0,0].plot(numeval, cond, color='#98D8D8', label = '$\kappa(\mathbf{C})$')
    ax[0,0].plot(numeval, sigma, color='#F8D030', label = '$\sigma$')
    ax[0,0].plot(numeval, psL2,  color='k', label = '$|| \mathbf{p}_{\sigma} ||$')
    if ( (idx == 2) or (live == False) ):
        ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1, fontsize = 8)

    # Upper Right Plot
    objVec = []
    if (plot_mean):
        ax[0,1].set_title('Mean of Objective Variables')
        objVec = mu
    else:
        ax[0,1].set_title('Objective Variables')
        objVec = fvalXvec

    ax[0,1].grid(True)
    for i in range(numdim):
        ax[0,1].plot(numeval, objVec[i], color = colors[i], label=names[i])
    if ( (idx == 2) or (live == False) ):
        ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

    # Lower Right Plot
    ax[1,0].set_title('Square Root of Eigenvalues of $\mathbf{C}$')
    ax[1,0].grid(True)
    ax[1,0].set_yscale('log')
    for i in range(numdim):
        ax[1,0].plot(numeval, axis[i], color = colors[i])

    # Lower Left Plot
    ax[1,1].set_title('$\sigma \sqrt{diag(\mathbf{C})}$')
    ax[1,1].grid(True)
    ax[1,1].set_yscale('log')
    for i in range(numdim):
        ax[1,1].plot(numeval, ssdev[i], color = colors[i], label=names[i])
