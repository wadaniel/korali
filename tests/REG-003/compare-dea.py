#!/usr/bin/env python3
import os
import sys
import json
from math import isclose

sys.path.append('./helpers')
from reg003_helpers import *

src_py  = "../REG-001/tutorials/a1-optimization/python/_result_run-dea/"
src_cxx = "../REG-002/tutorials/a1-optimization/cxx/_result_run-dea/"

resultfiles_py  = sorted( [f for f in os.listdir(src_py) if os.path.isfile(os.path.join(src_py, f))] )
resultfiles_cxx = sorted( [f for f in os.listdir(src_cxx) if os.path.isfile(os.path.join(src_cxx, f))] )

result_py  = resultfiles_py[-1]
result_cxx = resultfiles_cxx[-1]

json_py  = open_json(src_py, result_py)
json_cxx = open_json(src_cxx, result_cxx)

###############################################################################

# Compare Python and CXX results

assert_value( [ "Solver", "Internal", "Best Ever Sample" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Best Ever Value" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Current Best Sample" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Fitness Vector" ], json_cxx, json_py)
assert_value( [ "Solver", "Internal", "Sample Candidate" ], json_cxx, json_py)
