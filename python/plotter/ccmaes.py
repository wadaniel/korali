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
import matplotlib.transforms as transforms
from matplotlib.patches import Circle, Ellipse

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


# Plot CCMAES results (read from .json files)
def plot_ccmaes(src, live=False, test=False, evolution=False):

    live = live or evolution

    runid    = -1 # for safety check
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
    axis     = [] # sqrt(EVals)
    ssdev    = [] # sigma x diag(C)
    cov      = [] # covariance matrix (for evolution plot)
    mu_x     = [] # current mean dim 1 (for evolution plot)
    mu_y     = [] # current mean dim 2 (for evolution plot)
    
    ccmaes   = False
    via      = None
    normal   = None

    # Temporary Problem definition
    x = np.linspace(-32, 32, 1000)
    y = np.linspace(-32, 32, 1000)
    X, Y = np.meshgrid(x, y)
    Z = (np.square(1-X)+100*(np.square(Y+3-np.square(X))))

    plt.style.use('seaborn-dark')
 
    resultfiles = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    resultfiles = sorted(resultfiles)

    fig = None
    ax  = None
 
    if (resultfiles == []):
        print("[Korali] Error: Did not find file {0} in the result folder...".format(src))
        exit(-1)


    for filename in resultfiles:
        path   = '{0}/{1}'.format(src, filename)
        
        with open(path) as f:

            data       = json.load(f)
            solverName = data['Solver']

            state = data[solverName]['Internal']
            gen   = data['Current Generation']

            if (runid == -1):

                runid  = data['Run ID']
                numdim = len(data['Variables'])
                names  = [ data['Variables'][i]['Name'] for i in range(numdim) ]
                colors = hls_colors(numdim)
                
                if ( (evolution == True) and (numdim != 2) ):
                    print("[Korali] Error: Evolution feature only for 2D available - Bye!")
                    exit(0)
                else:
                    cov.append(state['Covariance Matrix'])
                
                # TODO check constraints
                #ccmaes = True
                #via    = [state['Viability Boundaries'][0]]
                #normal = [state['Constraint Normal Approximation']]
 
                for i in range(numdim):
                    fvalXvec.append([])
                    axis.append([])
                    ssdev.append([])

                if (live == True):
                    if (evolution == False):
                        fig, ax = plt.subplots(2,2,num='{0} live diagnostics'.format(solverName), figsize=(8,8))
                    else:
                        fig, ax = plt.subplots(1,1,num='CCMAES Evolution: {0}'.format(src), figsize=(8,8))

                    fig.show()

            if ( (live == True) and (not plt.fignum_exists(fig.number))):
                print("[Korali] Figure closed - Bye!")
                exit(0)
    

            if (data['Run ID'] != runid):
                print("[Korali] Warning: Skipping file {0}, results origin from" \
                        "a different experiment (different run id).".format(path))
                continue

            if gen > 1:

                numeval.append(data['Function Evaluation Count'])
                dfval.append(abs(state['Current Best Value'] - state['Best Ever Value']))
                
                fval.append(state['Current Best Value'])
                sigma.append(state['Sigma'])
                cond.append(state['Maximum Covariance Eigenvalue']/state['Minimum Covariance Eigenvalue'])
                psL2.append(state['Conjugate Evolution Path L2 Norm'])
                cov.append(state['Covariance Matrix'])

                if (evolution == True):
                    mu_x.append(state['PreviousMeanVector'][0])
                    mu_y.append(state['PreviousMeanVector'][1])
                    samples_x = [sublist[0] for sublist in state['Samples']]
                    samples_y = [sublist[1] for sublist in state['Samples']]
                
                if (ccmaes == True):
                    via.append(state['Viability Boundaries'][0])
                    normal.append(state['Constraint Normal Approximation'])

                for i in range(numdim):
                    fvalXvec[i].append(state['Current Best Sample'][i])
                    axis[i].append(state['Axis Lengths'][i])
                    ssdev[i].append(sigma[-1]*np.sqrt(state['Covariance Matrix'][i*numdim+i]))
            
                if (live == True and gen > 1):
                    if (evolution == False):
                        draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, axis, ssdev, colors, names, live)
                    else:
                        plt.clf()
                        fig, ax = plt.subplots(1,1,num='CCMAES Evolution: {0}'.format(src), figsize=(8,8))
                        draw_figure_evolution(fig, ax, src, gen, sigma, cov, mu_x, mu_y, samples_x, samples_y, ccmaes, normal, via, X, Y, Z)

                    plt_pause_light(0.05)

    if (live == False):
        fig, ax = plt.subplots(2,2,num='{0} live diagnostics'.format(solverName), figsize=(8,8))
        draw_figure(fig, ax, src, gen, numeval, numdim, fval, dfval, cond, sigma, psL2, fvalXvec, axis, ssdev, colors, names, live)
        fig.show()
    
    if (test == False):
        plt.pause(3600)
    
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


# Plot CCMAES samples, proposals, and mean (only 2D, read from .json files)
def draw_figure_evolution(fig, ax, src, idx, sigma, cov, mu_x, mu_y, samples_x, samples_y, ccmaes, normal, via, X, Y, Z):
 
    plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)),\
                  fontweight='bold',\
                  fontsize=12 )
    
    lambda_, v = np.linalg.eig(cov[-2])
    w = 5*sigma[-1]*lambda_[0]
    h = 5*sigma[-1]*lambda_[1]
    ang=np.rad2deg(np.arccos(v[0, 0]))

    if ccmaes == True:
        circle1  = Circle( (0, -2), radius = 1, facecolor = 'None', edgecolor='red', linewidth=2, label = 'Constraint Boundary' )
        circle2  = Circle( (0, -2), radius = np.sqrt(via[-2]+1), facecolor = 'None', edgecolor='red', linestyle='dashed', label = 'Viability Boundary' )
    ellipse = Ellipse( (mu_x[-1], mu_y[-1]), width=w, height=h, angle=ang, facecolor = 'None', edgecolor='b', label = 'Proposal Distribution' )

    ax.contour(X, Y, Z, [ 5, 55, 105, 155, 205, 255, 305], colors='#34495e', linewidths=1 )
    ax.set_xlim(-32,32)
    ax.set_ylim(-32,32)
    ax.plot(1,-2,'*', color = 'g', label = 'Minimum')
    ax.plot(samples_x,samples_y,'x', color = 'k', label = 'Samples')
    ax.plot(mu_x,mu_y,'^:', color = 'c', label = 'Historical Means' )
    ax.plot(mu_x[-1],mu_y[-1],'^', color = 'b', label = 'Current Mean')
    if ccmaes == True:
        ax.arrow( mu_x[-1], mu_y[-1], normal[-2][0][0],normal[-2][0][1],
                head_width=0.05, head_length=0.1, label = 'Constraint Normal Approximation' )
        ax.add_patch(circle1)
        ax.add_patch(circle2)
    ax.add_patch(ellipse)

    #plt_pause_light(1)
    #plt.savefig('ccmaes{0}.png'.format(idx), bbox_inches='tight')
    #if(live == False): time.sleep(0.1)
