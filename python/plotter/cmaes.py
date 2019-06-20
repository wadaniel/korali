#! /usr/bin/env python3

import os
import sys
import glob
import time
import json
import colorsys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from korali.plotter.helpers import plt_pause_light, plt_multicolored_lines

# Get a list of evenly spaced colors in HLS huse space.
# Credits: seaborn package
def hls_colors(num, h = 0.01, l=0.6, s=0.65):
    hues = np.linspace(0, 1, num + 1)[:-1]
    hues += h
    hues %= 1
    hues -= hues.astype(int)
    palette = [ list(colorsys.hls_to_rgb(h_i, l, s)) for h_i in hues ]
    return palette


# Get a list of strings for json keys of current results or best ever results
def objstrings(obj='current'):
    if obj == 'current':
        return ['CurrentBestFunctionValue', 'CurrentBestVector']
    elif obj == 'ever':
        return ['BestEverFunctionValue', 'BestEverVector']
    else:
        raise ValueError("obj must be 'current' or 'ever'")


# Plot CMA-ES results (read from .json files)
def plot_cmaes(src, live = False, obj='current'):

    numdim = 0 # problem dimension
    names    = [] # description params
    colors   = [] # rgb colors
    numeval  = [] # number obj function evaluations
    sigma    = [] # scaling parameter
    cond     = [] # condition of C (largest EW / smallest EW)
    psL2     = [] # conjugate evolution path L2 norm
    dfval    = [] # abs diff currentBest - bestEver
    fval     = [] # best fval current generation
    fvalXvec = [] # location fval
    axis     = [] # sqrt(EVals)
    ssdev    = [] # sigma x diag(C)

    plt.style.use('seaborn-dark')
 
    resultfiles = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    resultfiles = sorted(resultfiles)

    fig = None
    ax  = None
 
    if (resultfiles == []):
        print("[Korali] Error: Did not find file {0} in the _korali_result folder...".format(src))
        exit(-1)


    if (True):
        for filename in resultfiles:
            path   = '{0}/{1}'.format(src, filename)
            
            with open(path) as f:
 
                data  = json.load(f)
                solver = ''
                if ( 'CMA-ES' in data['Solver'] ):
                    solver = 'CMA-ES'
                elif ('CCMA-ES' in data['Solver'] ):
                    solver = 'CCMA-ES'
                else:
                    print("[Korali] Error: could not identify solver from result files!")
                    exit(-1)

                state = data[solver]['State']
                gen   = state['Current Generation']

                if (fig, ax) == (None, None):
 
                    numdim = len(data['Variables'])
                    names  = [ data['Variables'][i]['Name'] for i in range(numdim) ]
                    colors = hls_colors(numdim)
 
                    for i in range(numdim):
                        fvalXvec.append([])
                        axis.append([])
                        ssdev.append([])

                    if (live == True):
                        fig, ax = plt.subplots(2,2,num='{0} live diagnostics'.format(solver), figsize=(8,8))
                        fig.show()
 
                if (live == True and (not plt.fignum_exists(fig.number))):
                    print("[Korali] Figure closed - Bye!")
                    exit(0)

                if gen > 0:

                    numeval.append(state['EvaluationCount'])
                    dfval.append(abs(state["CurrentBestFunctionValue"] - state["BestEverFunctionValue"]))
                    
                    fval.append(state[objstrings(obj)[0]])
                    sigma.append(state['Sigma'])
                    cond.append(state['MaxEigenvalue']/state['MinEigenvalue'])
                    psL2.append(state['ConjugateEvolutionPathL2'])

                    for i in range(numdim):
                        fvalXvec[i].append(state[objstrings(obj)[1]][i])
                        axis[i].append(state['AxisLengths'][i])
                        ssdev[i].append(sigma[-1]*np.sqrt(state['CovarianceMatrix'][i][i]))
                
                    if (live == True and gen > 1):
                        draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, axis, ssdev, colors, names, live)

    if (live == False):
        fig, ax = plt.subplots(2,2,num='{0} live diagnostics'.format(solver), figsize=(8,8))
        fig.show()
        draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, axis, ssdev, colors, names, live)
        print("[Korali] Figure closed - Bye!")
        exit(0)


# Create Plot from Data
def draw_figure(fig, ax, src, idx, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, axis, ssdev, colors, names, live):

    plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)),\
                      fontweight='bold',\
                      fontsize=12 )

    # Upper Left Plot
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')
    plt_multicolored_lines(ax[0,0], numeval, fval, 0.0, 'r', 'b', '$| F |$')
    ax[0,0].plot(numeval, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
    ax[0,0].plot(numeval, cond, color='#98D8D8', label = '$\kappa(\mathbf{C})$')
    ax[0,0].plot(numeval, sigma, color='#F8D030', label = '$\sigma$')
    ax[0,0].plot(numeval, psL2,  color='k', label = '$|| \mathbf{p}_{\sigma} ||$')
    if ( (idx == 2) or (live == False) ):
        ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1, fontsize = 8)

    # Upper Right Plot
    ax[0,1].set_title('Objective Variables')
    ax[0,1].grid(True)
    for i in range(numdim):
        ax[0,1].plot(numeval, fvalXvec[i], color = colors[i], label=names[i])
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
    
    if (live == True):
        plt_pause_light(0.05)
    else:
        plt.pause(3600) #fix this (DW)


