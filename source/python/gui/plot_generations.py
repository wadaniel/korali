#! /usr/bin/env python3

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

resultdir = 'korali00000'

numeval  = []
fval     = []
sigma    = []
fvalXvec = []

plt.figure('CMA-ES live diagonistics: {0}'.format(resultdir),figsize=(8,8))
#fig, axes = plt.subplots(2, 2, sharex='all')

idx = 1
while( os.path.isfile( 'korali00000/s{1}.json'.format(resultdir, str(idx).zfill(5))) ):

    path = 'korali00000/s{0}.json'.format(str(idx).zfill(5))
    
    with open(path) as f:
        data = json.load(f)
        state = data['Solver']['State']

        numeval.append(data['Solver']['Lambda']*idx)
        fval.append(state['CurrentFunctionValue'])  # or maybe BestEver
        fvalXvec.append(state['CurrentBestVector']) # or maybe BestEverVector
        sigma.append(state['Sigma'])

    print(idx)

    plt.subplot(221)
    plt.title('Sigma')
    plt.grid(True)
    plt.yscale('log')
    plt.plot(numeval, sigma, color='g')
    plt.plot(numeval, fval, color='r')

    plt.subplot(222)
    plt.title('Object Variables')
    plt.grid(True)
    plt.plot(numeval, fvalXvec)
    # TODO: add labes

    plt.subplot(223)
    plt.title('Scaling (All Main Axes)')
    plt.grid(True)
    plt.yscale('log')
    plt.plot(numeval, numeval) # TODO
    
    plt.subplot(224)
    plt.title('Standard Deviation in All Coordinates')
    plt.grid(True)
    plt.yscale('log')
    plt.plot(numeval, numeval) # TODO
    
    plt.pause(0.05) 
    time.sleep(2)
    idx = idx+1

plt.show()
