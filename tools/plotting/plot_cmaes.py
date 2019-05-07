#! /usr/bin/env python3

import os
import sys
import glob
import time
import json
import colorsys
import numpy as np

import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt


from plot_helpers import plt_pause_light

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
def run_diagnostics(src, live = False, obj='current'):

    idx    = 0 # generation
    numdim = 0 # problem dimension

    names    = [] # description params
    colors   = [] # rgb colors
    numeval  = [] # number obj function evaluations
    numevalp = [] # number obj function evaluations for positive fval
    numevaln = [] # number obj function evaluations for negative fval
    sigma    = [] # scaling parameter
    cond     = [] # condition of C (largest EW / smallest EW)
    psL2     = [] # conjugate evolution path L2 norm
    dfval    = [] # abs diff currentBest - bestEver
    fvalneg  = [] # best fval current generation (fval < 0)
    fvalpos  = [] # best fval current generation (fval > 0)
    fvalXvec = [] # location fval
    axis     = [] # sqrt(EVals)
    ssdev    = [] # sigma x diag(C)

    plt.style.use('seaborn-dark')
    fig, ax = plt.subplots(2,2,num='CMA-ES live diagnostics: {0}'.format(src),figsize=(11,11))
    fig.show()

    updt_text = plt.figtext(0.5, 0.01, 'TEST', horizontalalignment='center');

    while( plt.fignum_exists(fig.number) ):

        if ( not os.path.isfile('{0}/s{1}.json'.format(src, str(idx).zfill(5))) ):
            if ( live == True ):
              if( idx > 0): plt_pause_light(0.5)
              continue
            else:
                break

        time.sleep(0.1) # XXX ok for now but needs better fix

        path = '{0}/s{1}.json'.format(src, str(idx).zfill(5))

        localtime = time.localtime(time.time())

        plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)),\
                      fontweight='bold',\
                      fontsize=12 )

        tmp_str = 'last update:{0}:{1}:{2}'.format(\
                          str(localtime.tm_hour).zfill(2),\
                          str(localtime.tm_min).zfill(2),\
                          str(localtime.tm_sec).zfill(2))

        updt_text.set_text( tmp_str )

        # plt.figtext(0.05,0.00, date_str, fontsize=14 )

        # plt.suptitle('s{0}.json\n(last update:{1}:{2}:{3})'.format(\
        #             str(idx).zfill(5),\
        #             str(localtime.tm_hour).zfill(2),\
        #             str(localtime.tm_min).zfill(2),\
        #             str(localtime.tm_sec).zfill(2)), fontsize=12)

        with open(path) as f:
            data  = json.load(f)
            state = data['Solver']['State']

            if idx == 0:
                numdim = data['Dimension']
                names  = [ data['Parameters'][i]['Name'] for i in range(numdim) ]
                colors = hls_colors(numdim)
                for i in range(numdim):
                    fvalXvec.append([])
                    axis.append([])
                    ssdev.append([])

                idx = idx + 1
                continue

            numeval.append(state['EvaluationCount'])
            f = state[objstrings(obj)[0]]
            dfval.append(abs(state["CurrentBestFunctionValue"] - state["BestEverFunctionValue"]))
            if f > 0 :
                if ( (not numevalp) & (len(numevaln) > 0) ):
                    # trick for conintuous plot
                    numevalp.append(numevaln[-1])
                    fvalpos.append(fvalneg[-1])
                fvalpos.append(f)
                numevalp.append(numeval[-1])
            else :
                if ( (not numevaln) & (len(numevalp) > 0) ):
                    # trick for conintuous plot
                    numevaln.append(numevalp[-1])
                    fvalneg.append(fvalpos[-1])
                fvalneg.append(f)
                numevaln.append(numeval[-1])
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
        if len(numevalp) > 0 : ax[0,0].plot(numevalp, fvalpos,  color='b', label = '| F |')
        if len(numevaln) > 0 : ax[0,0].plot(numevaln, [abs(v) for v in fvalneg], color='r', label = '| F |')
        ax[0,0].plot(numeval, dfval, 'x', color = '#34495e', label = '| F - F_best |')
        ax[0,0].plot(numeval, cond, color='#98D8D8', label = 'Condition')
        ax[0,0].plot(numeval, sigma, color='#F8D030', label = 'Sigma')
        ax[0,0].plot(numeval, psL2,  color='k', label = '|| Path ||')


        if idx == 2:
            ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1)

        ax[0,1].set_title('Objective Variables')
        ax[0,1].grid(True)
        for i in range(numdim):
            ax[0,1].plot(numeval, fvalXvec[i], color = colors[i], label=names[i])

        if idx == 2:
            ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)


        ax[1,0].set_title('Scaling (All Main Axes)')
        ax[1,0].grid(True)
        ax[1,0].set_yscale('log')
        for i in range(numdim):
            ax[1,0].plot(numeval, axis[i], color = colors[i])


        ax[1,1].set_title('Standard Deviation in All Coordinates')
        ax[1,1].grid(True)
        ax[1,1].set_yscale('log')
        for i in range(numdim):
            ax[1,1].plot(numeval, ssdev[i], color = colors[i], label=names[i])

        plt_pause_light(0.05)
        if(live == False): time.sleep(0.5)
        idx = idx+1

    fig.show()


if __name__ == '__main__':

    if (len(sys.argv) == 2):
        print  ("Plotting results from dir " + sys.argv[1])
        run_diagnostics(src=sys.argv[1], live = True)

    else:
        print("Invalid arguments, exit ...")
        print("(one argument required, directoy of CMA-ES results)")
