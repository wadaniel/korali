#! /usr/bin/env python3

import os
import json
import time
import matplotlib.pyplot as plt


outdir = 'korali00000/'
path = 'korali00000/s00035.json'

idx = 0

fval     = []
sigma    = []
bestXvec = []

while( os.path.isfile( 'korali00000/s{0}.json'.format(str(idx).zfill(5))) ):

    with open(path) as f:
        data = json.load(f)
        state = data['Solver']['State']

        fval.append(state['CurrentFunctionValue'])  # or maybe BestEver
        bestXvec.append(state['CurrentBestVector']) # or maybe BestEverVector
        sigma.append(state['Sigma'])

    idx = idx+1
    print(idx)
    print(sigma)
    print(bestXvec)


    time.sleep(2)

