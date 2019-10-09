#!/usr/bin/env python3
import numpy as np

def checkMin(k, expectedMinimum, tol):
    minimum = k["Solver"]["Internal"]["Best Ever Value"]
    assert np.isclose(expectedMinimum, minimum, atol = tol), "Minimum {0} "\
            "deviates from true min by more than {1}".format(minimum, tol)
