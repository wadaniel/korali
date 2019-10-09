#!/usr/bin/env python3
import json
import numpy as np

def open_json(dirname, filename):
    path = '{0}/{1}'.format(dirname, filename)
    data = None
    with open(path) as f:
       data = json.load(f)
    return data

def checkMean(k, expectedMean, tol):
    samples = k["Solver"]["Internal"]["Sample Database"]
    
    mean = np.mean(samples)
    assert np.isclose(expectedMean, mean, atol = tol), "Mean of Samples {0} "\
            "deviate from true mean by {1}".format(mean, tol)

def checkStd(k, expectedStd, tol):
    samples = k["Solver"]["Internal"]["Sample Database"]
    
    std = np.std(samples)
    assert np.isclose(expectedStd, std, atol = tol), "Standard Deviation of "\
        "Samples {0} deviate from true Std by {1}".format(std, tol)
