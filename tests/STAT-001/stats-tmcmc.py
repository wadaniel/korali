#!/usr/bin/env python3
import os
import sys
import json
import numpy as np

sys.path.append('./helpers')
from stat001_helpers import *

src_cxx = "../../tutorials/cxx/_a2_sampling_tmcmc/"
src_py  = "../../tutorials/python/_a2_sampling_tmcmc/"

def checkMean(js, expectedMean, tol):
    samples = js["TMCMC"]["Internal"]["Sample Parameters Database"]
    
    mean = np.mean(samples)
    assert np.isclose(expectedMean, mean, atol = tol), "Mean of Samples {0}"\
            "deviate from true mean by {1}".format(mean, tol)

def checkStd(js, expectedStd, tol):
    samples = js["TMCMC"]["Internal"]["Sample Parameters Database"]
    
    std = np.std(samples)
    assert np.isclose(expectedStd, std, atol = tol), "Standard Deviation of "\
        "Samples {0} deviate from true Std by {1}".format(std, tol)


resultfiles_cxx = sorted( [f for f in os.listdir(src_cxx) if os.path.isfile(os.path.join(src_cxx, f))] )
resultfiles_py  = sorted( [f for f in os.listdir(src_py) if os.path.isfile(os.path.join(src_py, f))] )

result_cxx = resultfiles_cxx[-1]
result_py  = resultfiles_py[-1]

json_cxx = open_json(src_cxx, result_cxx)
json_py  = open_json(src_py, result_py)

# Check CXX results
checkMean(json_cxx, 0.0, 0.05)
checkStd(json_cxx, 1.0, 0.5)

# Check python results
checkMean(json_py, 0.0, 0.05)
checkStd(json_py, 1.0, 0.5)
