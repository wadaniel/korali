#!/usr/bin/env python3
import os
import sys
import json
from math import isclose

sys.path.append('./helpers')
from reg003_helpers import *

src_cxx = "../../tutorials/cxx/_a1_optimization_dea_result/"
src_py  = "../../tutorials/python/_a1_optimization_dea_result/"

resultfiles_cxx = sorted( [f for f in os.listdir(src_cxx) if os.path.isfile(os.path.join(src_cxx, f))] )
resultfiles_py  = sorted( [f for f in os.listdir(src_py) if os.path.isfile(os.path.join(src_py, f))] )

result_cxx = resultfiles_cxx[-1]
result_py  = resultfiles_py[-1]

json_cxx = open_json(src_cxx, result_cxx)
json_py  = open_json(src_py, result_py)

###############################################################################

# Compare Python and CXX results

assert_value( [ "DEA", "Internal", "Best Ever Sample" ], json_cxx, json_py)

assert_value( [ "DEA", "Internal", "Best Ever Value" ], json_cxx, json_py)

assert_value( [ "DEA", "Internal", "Current Best Sample" ], json_cxx, json_py)

assert_value( [ "DEA", "Internal", "Fitness Vector" ], json_cxx, json_py)

assert_value( [ "DEA", "Internal", "Sample Candidate" ], json_cxx, json_py)
