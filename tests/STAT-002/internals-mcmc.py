#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
from math import isclose

def open_json(dirname, filename):
    path = '{0}/{1}'.format(dirname, filename)
    data = None
    with open(path) as f:
       data = json.load(f)
    return data

def compareMean(js):
    samples = js["Solver"]["Internal"]["Sample Database"]
    
    mean      = np.mean(samples)
    chainmean = js["Solver"]["Internal"]["Chain Mean"]
    assert np.isclose(mean, chainmean), "Chain Mean deviates from Mean of "\
            "Samples"

def compareStd(js):
    samples = js["Solver"]["Internal"]["Sample Database"]

    mean     = np.mean(samples)
    std      = np.sqrt(sum((samples - mean)**2)/(len(samples)-1))
    chainstd = js["Solver"]["Internal"]["Cholesky Decomposition Chain Covariance"]
    assert np.isclose(std, chainstd), "Cholesky Decomposition of Chain" \
            "Covariance deviates from Standard Deviation of Samples"

src  = "_result_run-mcmc"
json  = open_json(src, 'final.json')
compareMean(json)
compareStd(json)
