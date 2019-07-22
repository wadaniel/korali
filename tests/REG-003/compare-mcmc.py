#!/usr/bin/env python3
import os
import sys
import json
from math import isclose

sys.path.append('./helpers')
from reg003_helpers import *

src_py  = "../REG-001/tutorials/a2-sampling/python/_result_run-mcmc/"
src_cxx = "../REG-002/tutorials/a2-sampling/cxx/_result_run-mcmc/"

resultfiles_py  = sorted( [f for f in os.listdir(src_py) if os.path.isfile(os.path.join(src_py, f))] )
resultfiles_cxx = sorted( [f for f in os.listdir(src_cxx) if os.path.isfile(os.path.join(src_cxx, f))] )

result_py  = resultfiles_py[-1]
result_cxx = resultfiles_cxx[-1]

json_py  = open_json(src_py, result_py)
json_cxx = open_json(src_cxx, result_cxx)

###############################################################################

# Compare Python and CXX results

assert_value( [ "General", "Random Seed" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Acceptance Count" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Acceptance Rate" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Chain Covariance" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Chain Mean" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Sample Fitness Database" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Rejection Count" ], json_cxx, json_py)

