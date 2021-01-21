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
    generations = np.zeros((N,))

    for index, key in enumerate(genList.keys()):
        weights[index] = genList[key]['Problem']['Weights']
        means[index] = genList[key]['Problem']['Means']
        covariances[index] = genList[key]['Problem']['Covariances']
        generations[index] = key

    fig, ax = plt.subplots(3,M)
    for m in range(M):
        ax[0,m].plot(generations,weights[:,m])
        ax[0,m].grid()
        ax[1,m].plot(generations,means[:,m,:])
        ax[1,m].grid()
        ax[2,m].plot(generations,covariances[:,m,:])
        ax[2,m].grid()
    plt.show()
