#! /usr/bin/env python3

import os
import sys
import glob
import time
import json
import colorsys
import numpy as np
import matplotlib.pyplot as plt

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
def run_diagnostics(live, src, obj='current'):
     
    idx    = 0 # generation
    numdim = 0 # problem dimension

    colors   = [] # rgb colors
    numeval  = [] # number obj function evaluations
    sigma    = [] # scaling parameter
    cond     = [] # condition of C (largest EW / smallest EW)
    fval     = [] # best fval current generation
    fvalXvec = [] # location fval
    axis     = [] # sqrt(EVals)
    Csdev    = [] # sigma x diag(C)

    plt.style.use('seaborn-dark')
    fig = plt.figure('CMA-ES live diagnostics: {0}'.format(src),figsize=(8,8))
       
    while( plt.fignum_exists(fig.number) ):

        if ( not os.path.isfile('{0}/s{1}.json'.format(src, str(idx).zfill(5))) ):
            if ( live == True):    
                plt.pause(0.05)
                continue
            else: 
                break

        path = '{0}/s{1}.json'.format(src, str(idx).zfill(5))

        localtime = time.localtime(time.time())
        plt.suptitle('s{0}.json\n(last update:{1}:{2}:{3})'.format(str(idx).zfill(5),\
                    str(localtime.tm_hour).zfill(2),\
                    str(localtime.tm_min).zfill(2),\
                    str(localtime.tm_sec).zfill(2)), fontsize=14)

        with open(path) as f:
            data = json.load(f)
            state = data['Solver']['State']

            if idx == 0:
                numdim = len(state['AxisLengths'])
                colors = hls_colors(numdim)
                for i in range(numdim):
                    fvalXvec.append([])
                    axis.append([])
                    Csdev.append([])

                idx = idx + 1
                continue

            numeval.append(state['EvaluationCount'])
            fval.append(state[objstrings(obj)[0]])
            sigma.append(state['Sigma'])
            cond.append(state['MaxEigenvalue']/state['MinEigenvalue'])

            for i in range(numdim):
                fvalXvec[i].append(state[objstrings(obj)[1]][i])
                axis[i].append(state['AxisLengths'][i])
                Csdev[i].append(sigma[idx-1]*np.sqrt(state['CovarianceMatrix'][i][i]))


        ax221 = plt.subplot(221)
        plt.grid(True)
        plt.yscale('log')
        plt.plot(numeval, sigma, color='#F8D030', label = 'Sigma')
        plt.plot(numeval, fval,  color='#C03028', label = 'FVal')
        plt.plot(numeval, cond,  color='#98D8D8', label = 'Cond')

        if idx == 1:
            plt.legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1)


        ax222 = plt.subplot(222)
        plt.title('Object Variables')
        plt.grid(True)
        for i in range(numdim):
            ax222.plot(numeval, fvalXvec[i], color = colors[i], label='X'+str(i+1))
        
        if idx == 1:
            plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

        ax223 = plt.subplot(223)
        plt.title('Scaling (All Main Axes)')
        plt.grid(True)
        plt.yscale('log')
        for i in range(numdim):
            ax223.plot(numeval, axis[i], color = colors[i])
        
        ax224 = plt.subplot(224)
        plt.title('Standard Deviation in All Coordinates')
        plt.grid(True)
        plt.yscale('log')
        for i in range(numdim):
            ax224.plot(numeval, Csdev[i], color = colors[i], label='X'+str(i+1))

        if idx == 1:
            plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

       
        plt.pause(0.05) 
        if(live == False): time.sleep(0.5)
        idx = idx+1

    plt.show()


if __name__ == '__main__':

    if (len(sys.argv) == 2):
        print  ("Plotting results from dir " + sys.argv[1])
        run_diagnostics(live=True, src=sys.argv[1])

    else: 
        print("Invalid arguments, exit ...")
        print("(one argument required, directoy of CMA-ES results)")
