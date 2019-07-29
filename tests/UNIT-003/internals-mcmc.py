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
    samples = js["Solver"]["Internal"]["Sample Parameters Database"]
    
    mean      = np.mean(samples)
    chainmean = js["Solver"]["Internal"]["Chain Mean"]
    assert np.isclose(mean, chainmean), "Chain Mean deviates from Mean of "\
            "Samples"

def compareStd(js):
    samples = js["Solver"]["Internal"]["Sample Parameters Database"]

    mean     = np.mean(samples)
    std      = np.sqrt(sum((samples - mean)**2)/(len(samples)-1))
    chainstd = js["Solver"]["Internal"]["Cholesky Decomposition Chain Covariance"]
    assert np.isclose(std, chainstd), "Cholesky Decomposition of Chain" \
            "Covariance deviates from Standard Deviation of Samples"

src_py  = "../REG-001/tutorials/a2-sampling/python/_result_run-mcmc/"
src_cxx = "../REG-002/tutorials/a2-sampling/cxx/_result_run-mcmc/"

resultfiles_py  = sorted( [f for f in os.listdir(src_py) if os.path.isfile(os.path.join(src_py, f))] )
resultfiles_cxx = sorted( [f for f in os.listdir(src_cxx) if os.path.isfile(os.path.join(src_cxx, f))] )

result_py  = resultfiles_py[-1]
result_cxx = resultfiles_cxx[-1]

json_py  = open_json(src_py, result_py)
json_cxx = open_json(src_cxx, result_cxx)

# Compare CXX
compareMean(json_cxx)
compareStd(json_cxx)

# Compare Python
compareMean(json_py)
compareStd(json_py)
