#!/bin/bash
import os
import sys
import json
import korali

sys.path.append('./helpers')
from reg006_helpers import *

#################################################
# Set Up  TMCMC run
#################################################

print("[Korali] Prepare TMCMC run")

# Starting Korali's Engine
k = korali.initialize()

k["Problem"]["Type"] = "Sampling"

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Covariance Scaling"] = 0.001

k["Solver"]["Default Burn In"] = 5
k["Solver"]["Burn In Steps"][0] = 10
k["Solver"]["Burn In Steps"][1] = 7

k["General"]["Random Seed"] = 314

k.setModel(evaluateModel)

#################################################
#  Run TMCMC
#################################################

print("[Korali] Run TMCMC..")
k.run()


#################################################
#  Prepare Expected Values
#################################################

gen = 0
currentBurnIn = [10, 7, 5]


#################################################
# Read Results
#################################################

print("[Korali] Read & Evaluate Output..")

src = "_korali_result"
resultfiles = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
resultfiles = sorted(resultfiles)
resultfiles = resultfiles[1:]

for filename in resultfiles:
  path   = '{0}/{1}'.format(src, filename)
        
  with open(path) as f:
    data  = json.load(f)

#################################################
# Test Results
#################################################

    print("[Korali] Checking generation {0} inside file {1} ..".format(gen, path))
    
    assert_value( data['Solver']['Default Burn In'], 5 )
    #assert_value( data['Solver']['Internal']['Current Burn In'], currentBurnIn[gen] )

    gen = gen + 1
