#!/usr/bin/env python3
import os
import sys
import numpy as np

def model_propagation(s):
    mu = s['Parameters'][0]
    var = s['Parameters'][1]

    print(np.random.normal(mu, var, 1))
