#!/usr/bin/env python3
import numpy as np

def checkMin(k, expectedMinimum, tol):
    minimum = k["Solver"]["Best Ever Value"]
    assert np.isclose(expectedMinimum, minimum, atol = tol), "Minimum {0} "\
            "deviates from true min {1} by more than {2}".format(minimum, expectedMinimum, tol)
