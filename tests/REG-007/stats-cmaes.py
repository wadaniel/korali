#!/usr/bin/env python3
import os
import sys
import json
import numpy as np

sys.path.append('./helpers')
from helpers import *

src  = "_results_cmaes"

def checkMin(js, expectedMinimum, tol):
    minimum = js["Solver"]["Internal"]["Best Ever Value"]
    assert np.isclose(expectedMinimum, minimum, atol = tol), "Minimum {0} "\
            "deviates from true min by more than {1}".format(minimum, tol)

result  = 'final.json'
json  = open_json(src, result)
checkMin(json, 0.22942553779431113, 1e-4)
