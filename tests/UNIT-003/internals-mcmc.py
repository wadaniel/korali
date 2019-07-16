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
    samples = js["MCMC"]["Internal"]["Sample Parameters Database"]
    
    mean      = np.mean(samples)
    chainmean = js["MCMC"]["Internal"]["Chain Mean"]
    assert np.isclose(mean, chainmean), "Chain Mean deviates from Mean of "\
            "Samples"

def compareStd(js):
    samples = js["MCMC"]["Internal"]["Sample Parameters Database"]

    mean     = np.mean(samples)
    std      = np.sqrt(sum((samples - mean)**2)/(len(samples)-1))
    chainstd = js["MCMC"]["Internal"]["Cholesky Decomposition of Chain Covariance"]
    assert np.isclose(std, chainstd), "Cholesky Decomposition of Chain " \
            "Covariance deviates from Standard Deviation of Samples"

src_cxx = "../../tutorials/cxx/_a2_sampling_mcmc_result/"
src_py  = "../../tutorials/python/_a2_sampling_mcmc_result/"

resultfiles_cxx = sorted( [f for f in os.listdir(src_cxx) if os.path.isfile(os.path.join(src_cxx, f))] )
resultfiles_py  = sorted( [f for f in os.listdir(src_py) if os.path.isfile(os.path.join(src_py, f))] )

result_cxx = resultfiles_cxx[-1]
result_py  = resultfiles_py[-1]

json_cxx = open_json(src_cxx, result_cxx)
json_py  = open_json(src_py, result_py)

# Compare CXX
compareMean(json_cxx)
compareStd(json_cxx)

# Compare Python
compareMean(json_py)
compareStd(json_py)
