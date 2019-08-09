#!/usr/bin/env python3
import os
import sys
import json
import numpy as np

sys.path.append('./helpers')
from helpers import *

src  = "_result_run-tmcmc/"

def checkMean(js, expectedMean, tol):
    samples = js["Solver"]["Internal"]["Sample Parameters Database"]
    
    mean = np.mean(samples)
    assert np.isclose(expectedMean, mean, atol = tol), "Mean of Samples {0}"\
            "deviate from true mean by {1}".format(mean, tol)

def checkStd(js, expectedStd, tol):
    samples = js["Solver"]["Internal"]["Sample Parameters Database"]
    
    std = np.std(samples)
    assert np.isclose(expectedStd, std, atol = tol), "Standard Deviation of "\
        "Samples {0} deviate from true Std by {1}".format(std, tol)


result  = 'final.json'
json  = open_json(src, result)
checkMean(json, 0.0, 0.05)
checkStd(json, 1.0, 0.5)
