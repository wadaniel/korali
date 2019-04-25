#! /usr/bin/env python3

import os
import json
import time
import numpy as np
import colorsys
import matplotlib.pyplot as plt

seaborn_avail = False

try:
    import seaborn as sns
    seaborn_avail = True
    sns.set_style("darkgrid")

except ImportError:
    print ("WARNING: package \'seaborn\' not available, using default plotting"
           "schemes (install with \'pip install seaborn\')")
    pass 

def hls_colors(num, h = 0.01, l=0.6, s=0.65):
    """Get a set of evenly spaced colors in HLS huse space.

    h, l, and s should be between 0 an 1


    Returns
    -------
    List of RGB tuples.

    
    Credits
    -------
    Code copied from seaborn package.

    """
    hues = np.linspace(0, 1, num + 1)[:-1]
    hues += h
    hues %= 1
    hues -= hues.astype(int)
    palette = [ list(colorsys.hls_to_rgb(h_i, l, s)) for h_i in hues ]
    return palette

resultdir = 'korali00000'

numeval  = []
sigma    = []
fval     = []
cond     = []
fvalXvec = []
axis     = []
Csdev    = []


fig = plt.figure('CMA-ES live diagnostics: {0}'.format(resultdir),figsize=(8,8))

idx    = 0
numdim = 0
colors = []

while( plt.fignum_exists(fig.number) & os.path.isfile( 'korali00000/s{1}.json'.format(resultdir, str(idx).zfill(5))) ):

    path = 'korali00000/s{0}.json'.format(str(idx).zfill(5))

    localtime = time.localtime(time.time())
    plt.suptitle('CMA-ES: s{0}.json (last update: {1}:{2}:{3})'.format(str(idx).zfill(5), localtime.tm_hour, localtime.tm_min, localtime.tm_sec), fontsize=14)

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
        fval.append(state['CurrentBestFunctionValue']) # or maybe BestEver
        sigma.append(state['Sigma'])
        cond.append(state['MaxEigenvalue']/state['MinEigenvalue'])

        for i in range(numdim):
            fvalXvec[i].append(state['CurrentBestVector'][i]) # or maybe BestEverVector
            axis[i].append(state['AxisLengths'][i]) # sqrt(Evals)
            Csdev[i].append(sigma[idx-1]*np.sqrt(state['CovarianceMatrix'][i][i]))


    plt.subplot(221)
    plt.title('FunctionValue (red), Sigma (yellow), Condition (blue)')
    plt.grid(True)
    plt.yscale('log')
    plt.plot(numeval, sigma, color='#F8D030')
    plt.plot(numeval, fval,  color='#C03028')
    plt.plot(numeval, cond,  color='#98D8D8')

    ax222 = plt.subplot(222)
    plt.title('Object Variables')
    plt.grid(True)
    for i in range(numdim):
        ax222.plot(numeval, fvalXvec[i], color = colors[i], label='X'+str(i+1))
    
    if idx == 1:
        legend = ax222.legend(loc='upper center')
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
        legend = ax224.legend(loc='upper center')
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

   
    plt.pause(0.05) 
    time.sleep(2)
    idx = idx+1

plt.show()
