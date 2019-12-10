#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine

# Plot CMAES results (read from .json files)
def plot(js):
    fig, ax = plt.subplots(2,2,num='Korali Results', figsize=(8,8))
    solver = js['Generations'][0]['Solver']['Type']
    numdim = len(js['Variables'])
    names  = [ js['Variables'][i]['Name'] for i in range(numdim) ]
    
    fval = []
    dfval = []
    gen = []
    sigma = []
    psL2 = []
    objVec = []
    evoweights = []

    for s in js['Generations']:
     fval.append(s['Solver']['Internal']['Current Best Value'])
     dfval.append(abs(s['Solver']['Internal']['Current Best Value'] - s['Solver']['Internal']['Best Ever Value']))
     gen.append(s['Internal']['Current Generation'])
     sigma.append(s['Solver']['Internal']['Sigma'])
     psL2.append(s['Solver']['Internal']['Conjugate Evolution Path L2 Norm'])
     objVec.append(s['Solver']['Internal']['Current Best Variables'])
     evoweights.append(s['Solver']['Internal']['Evolution Path Weights'])

    plt.suptitle('CMAES Diagnostics', fontweight='bold', fontsize=12 )

    # Upper Left Plot
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')
    drawMulticoloredLine(ax[0,0], gen, fval, 0.0, 'r', 'b', '$| F |$')
    ax[0,0].plot(gen, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
    ax[0,0].plot(gen, sigma, color='#F8D030', label = '$\sigma$')
    ax[0,0].plot(gen, psL2,  color='k', label = '$|| \mathbf{p}_{\sigma} ||$')
    
    #if ( (idx == 2) or (updateLegend == False) ):
    ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1, fontsize = 8)

    colors = hlsColors(numdim)
    
    # Upper Right Plot
    #if (plot_mean):
    #    ax[0,1].set_title('Mean of Objective Variables')
    #    objVec = mu
    #else:
    ax[0,1].set_title('Objective Variables')
    ax[0,1].grid(True)
    for i in range(numdim):
        ax[0,1].plot(gen, objVec, color = colors[i], label=names[i])
   
    #if ( (idx == 2) or (updateLegend == False) ):
    ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

    # Lower Right Plot
    ax[1,0].set_title('Evolution Path Weights')
    ax[1,0].grid(True)
    ax[1,0].set_yscale('log')
    for i in range(numdim):
        ax[1,0].plot(gen, evoweights, color = colors[i])

    # Lower Left Plot
    ax[1,1].set_title('$\sigma \sqrt{diag(\mathbf{C})}$')
    ax[1,1].grid(True)
    ax[1,1].set_yscale('log')
        
    plt.show()
