#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append( os.path.join( os.path.dirname(__file__), '..' ) )

def plot( genList, args ):
    numgens = len(genList)

    N = len(genList);
    i = list(sorted(genList.keys()))[0]
    M = genList[i]['Problem']['Number Of Distributions']
    dim = len(genList[i]['Problem']['Means'][0])

    weights = np.zeros((N,M))
    means = np.zeros((N,M,dim))
    covariances = np.zeros((N,M,dim*dim))
    loglikelihood = np.zeros((N,))
    generations = np.zeros((N,))

    for index, key in enumerate(genList.keys()):
        weights[index] = genList[key]['Problem']['Weights']
        means[index] = genList[key]['Problem']['Means']
        covariances[index] = genList[key]['Problem']['Covariances']
        loglikelihood[index] = genList[key]['Problem']['Data Loglikelihood']
        generations[index] = key

    fig, ax = plt.subplots(4,M)
    for m in range(M):
        ax[0,m].plot(generations,weights[:,m])
        ax[0,m].grid()
        ax[1,m].plot(generations,means[:,m,:])
        ax[1,m].grid()
        ax[2,m].plot(generations,covariances[:,m,:])
        ax[2,m].grid()

    ax[3,0].plot(generations,loglikelihood)
    ax[3,0].grid()
    for m in range(1,M):
        ax[3,m].axis('off')

    ax[0,0].set_ylabel(f'Weights')
    ax[1,0].set_ylabel(f'Means')
    ax[2,0].set_ylabel(f'Covariances')
    ax[3,0].set_ylabel('Data Loglikelihood')

    plt.show()
