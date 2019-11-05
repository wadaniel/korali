#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine

# Create Plot from Data
def plot(js):

    fig, ax = plt.subplots(2,2,num='Korali Results', figsize=(8,8))
    solver = js['Solvers'][0]['Type']
    numdim = len(js['Variables'])
    names  = [ js['Variables'][i]['Name'] for i in range(numdim) ]
    
    fval = []
    objVec = []
    dfval = []
    gen = []
    width = []
    means = []

    for s in js['Solvers']:
     objVec.append(s['Internal']['Current Best Variables'])
     fval.append(s['Internal']['Current Best Value'])
     dfval.append(abs(s['Internal']['Current Best Value'] - s['Internal']['Best Ever Value']))
     gen.append(s['Internal']['Current Generation'])
     width.append(s['Internal']['Max Distances'])
     means.append(s['Internal']['Current Mean'])

    plt.suptitle('DEA Diagnostics', fontweight='bold', fontsize=12 )
    colors = hlsColors(numdim)
    
    # Upper Left Plot
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')
    drawMulticoloredLine(ax[0,0], gen, fval, 0.0, 'r', 'b', '$| F |$')
    ax[0,0].plot(gen, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
    #if ( (idx == 2) or (updateLegend == False) ):
    ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1, fontsize = 8)

    # Upper Right Plot
    ax[0,1].set_title('Objective Variables')
    ax[0,1].grid(True)
    for i in range(numdim):
        ax[0,1].plot(gen, objVec, color = colors[i], label=names[i])
    #if ( (idx == 2) or (updateLegend == False) ):
    ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

    # Lower Right Plot
    ax[1,0].set_title('Width Population')
    ax[1,0].grid(True)
    for i in range(numdim):
        ax[1,0].plot(gen, width, color = colors[i])

    # Lower Left Plot
    ax[1,1].set_title('Mean Population')
    ax[1,1].grid(True)
    for i in range(numdim):
        ax[1,1].plot(gen, means, color = colors[i], label=names[i])
    #if ( (idx == 2) or (updateLegend == False) ):
    ax[1,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

    plt.show()

