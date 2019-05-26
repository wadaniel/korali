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

from korali.plotters.helpers import plt_pause_light, plt_multicolored_lines

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

    idx    = 0 # generation
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
    fig, ax = plt.subplots(2,2,num='CMA-ES live diagnostics: {0}'.format(src),figsize=(8,8))
    fig.show()


    while( plt.fignum_exists(fig.number) ):

        path = '{0}/s{1}.json'.format(src, str(idx).zfill(5))
        
        if ( not os.path.isfile(path) ):
            if ( live == True ):
              if( idx > 0): plt_pause_light(0.5)
              continue
            else:
                break

        time.sleep(0.1) # XXX ok for now but needs better fix (GA)

        localtime = time.localtime(time.time())

        plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)),\
                      fontweight='bold',\
                      fontsize=12 )
        
        #plt.text(x = 1000.65, y = 50, s = 'just sth', fontsize = 10, alpha = .85)

        tmp_str = 'last update:{0}:{1}:{2}'.format(\
                          str(localtime.tm_hour).zfill(2),\
                          str(localtime.tm_min).zfill(2),\
                          str(localtime.tm_sec).zfill(2))


        with open(path) as f:
            data  = json.load(f)
            state = data['Solver']['State']

            if idx == 0:
                dims    = data['Problem']['Variables']
                numdim  = len(dims)
                names  = [ data['Problem']['Variables'][i]['Name'] for i in range(numdim) ]
                colors = hls_colors(numdim)
                for i in range(numdim):
                    fvalXvec.append([])
                    axis.append([])
                    ssdev.append([])

                idx = idx + 1
                continue

            numeval.append(state['EvaluationCount'])
            dfval.append(abs(state["CurrentBestFunctionValue"] - state["BestEverFunctionValue"]))
            
            f = state[objstrings(obj)[0]]
            fval.append(f)
            sigma.append(state['Sigma'])
            cond.append(state['MaxEigenvalue']/state['MinEigenvalue'])
            psL2.append(state['ConjugateEvolutionPathL2'])

            for i in range(numdim):
                fvalXvec[i].append(state[objstrings(obj)[1]][i])
                axis[i].append(state['AxisLengths'][i])
                ssdev[i].append(sigma[idx-1]*np.sqrt(state['CovarianceMatrix'][i][i]))

        if idx < 2:
            idx = idx + 1
            continue


        ax[0,0].grid(True)
        ax[0,0].set_yscale('log')
        plt_multicolored_lines(ax[0,0], numeval, fval, 0.0, 'r', 'b', '$| F |$')
        ax[0,0].plot(numeval, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
        ax[0,0].plot(numeval, cond, color='#98D8D8', label = '$\kappa(\mathbf{C})$')
        ax[0,0].plot(numeval, sigma, color='#F8D030', label = '$\sigma$')
        ax[0,0].plot(numeval, psL2,  color='k', label = '$|| \mathbf{p}_{\sigma} ||$')


        if idx == 2:
            ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1, fontsize = 8)

        ax[0,1].set_title('Objective Variables')
        ax[0,1].grid(True)
        for i in range(numdim):
            ax[0,1].plot(numeval, fvalXvec[i], color = colors[i], label=names[i])

        if idx == 2:
            ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)


        ax[1,0].set_title('Square Root of Eigenvalues of $\mathbf{C}$')
        ax[1,0].grid(True)
        ax[1,0].set_yscale('log')
        for i in range(numdim):
            ax[1,0].plot(numeval, axis[i], color = colors[i])


        ax[1,1].set_title('$\sigma \sqrt{diag(\mathbf{C})}$')
        ax[1,1].grid(True)
        ax[1,1].set_yscale('log')
        for i in range(numdim):
            ax[1,1].plot(numeval, ssdev[i], color = colors[i], label=names[i])

        plt_pause_light(0.05)
        if(live == False): time.sleep(0.1)
        idx = idx+1

    fig.show()

