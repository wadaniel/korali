#!/usr/bin/env python3
import os
import sys
import json
from math import isclose

sys.path.append('./helpers')
from reg003_helpers import *

src_cxx = "../../tutorials/cxx/_a2_sampling_mcmc_result/"
src_py  = "../../tutorials/python/_a2_sampling_mcmc_result/"

resultfiles_cxx = sorted( [f for f in os.listdir(src_cxx) if os.path.isfile(os.path.join(src_cxx, f))] )
resultfiles_py  = sorted( [f for f in os.listdir(src_py) if os.path.isfile(os.path.join(src_py, f))] )

result_cxx = resultfiles_cxx[-1]
result_py  = resultfiles_py[-1]

json_cxx = open_json(src_cxx, result_cxx)
json_py  = open_json(src_py, result_py)

###############################################################################

# Compare Python and CXX results

assert_value( [ "MCMC", "Internal", "Acceptance Count" ], json_cxx, json_py)

assert_value( [ "MCMC", "Internal", "Acceptance Rate" ], json_cxx, json_py)

assert_value( [ "MCMC", "Internal", "Chain Covariance" ], json_cxx, json_py)

assert_value( [ "MCMC", "Internal", "Chain Mean" ], json_cxx, json_py)

assert_value( [ "MCMC", "Internal", "Sample Fitness Database" ], json_cxx, json_py)

assert_value( [ "MCMC", "Internal", "Rejection Count" ], json_cxx, json_py)

