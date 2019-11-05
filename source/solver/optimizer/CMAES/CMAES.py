#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, pauseLight, drawMulticoloredLine, checkFigure

# Plot CMAES results (read from .json files)
def plot(js):
    fig, ax = plt.subplots(2,2,num='Korali Results', figsize=(8,8))
    solver = js['Solvers'][0]['Type']
    numdim = len(js['Variables'])
    names  = [ js['Variables'][i]['Name'] for i in range(numdim) ]
    
    axis = []
    cond = []
    fval = []
    dfval = []
    gen = []
    ssdev = []
    sigma = []
    psL2 = []
    objVec = []

    for s in js['Solvers']:
     cond.append(s['Internal']['Maximum Covariance Eigenvalue']/s['Internal']['Minimum Covariance Eigenvalue'])
     fval.append(s['Internal']['Current Best Value'])
     dfval.append(abs(s['Internal']['Current Best Value'] - s['Internal']['Best Ever Value']))
     gen.append(s['Internal']['Current Generation'])
     sigma.append(s['Internal']['Sigma'])
     psL2.append(s['Internal']['Conjugate Evolution Path L2 Norm'])
     axis.append(s['Internal']['Axis Lengths'])
     objVec.append(s['Internal']['Current Best Variables'])


    ssdev = [ [ ] for i in range(numdim) ]
    for i in range(numdim):
     for s in js['Solvers']:
      ssdev[i].append(js['Solvers'][-1]["Internal"]["Sigma"]*np.sqrt(s["Internal"]['Covariance Matrix'][i*numdim+i]))
     
         
    plt.suptitle('CMAES Diagnostics', fontweight='bold', fontsize=12 )

    # Upper Left Plot
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')
    drawMulticoloredLine(ax[0,0], gen, fval, 0.0, 'r', 'b', '$| F |$')
    ax[0,0].plot(gen, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
    ax[0,0].plot(gen, cond, color='#98D8D8', label = '$\kappa(\mathbf{C})$')
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
    ax[1,0].set_title('Square Root of Eigenvalues of $\mathbf{C}$')
    ax[1,0].grid(True)
    ax[1,0].set_yscale('log')
    for i in range(numdim):
        ax[1,0].plot(gen, axis, color = colors[i])

    # Lower Left Plot
    ax[1,1].set_title('$\sigma \sqrt{diag(\mathbf{C})}$')
    ax[1,1].grid(True)
    ax[1,1].set_yscale('log')
    for i in range(numdim):
        ax[1,1].plot(gen, ssdev[i], color = colors[i], label=names[i])
        
    pauseLight(0.05)
    plt.show()