#!/bin/bash
import os
import sys
import json
import korali

sys.path.append('./helpers')
from helpers import *


#################################################
#  Prepare Expected Values
#################################################

gen = 0
currentBurnIn = [10, 7, 5, 5, 5, 5, 5, 5, 5]

eps = 1e-12
prevRho = 0.0
minRhoUpdate = 1e-3
maxRhoUpdate = 0.2


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
k["Solver"]["Max Chain Length"] = 1
k["Solver"]["Target Coefficient Of Variation"] = 0.5
k["Solver"]["Min Annealing Exponent Update"] = minRhoUpdate
k["Solver"]["Max Annealing Exponent Update"] = maxRhoUpdate

k["General"]["Random Seed"] = 314

k.setDirectModel(evaluateModel)


#################################################
#  Run TMCMC
#################################################

print("[Korali] Run TMCMC..")
k.run()


#################################################
# Read Result Files
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
# Test Result
#################################################

    print("[Korali] Checking generation {0} inside file {1} ..".format(gen, path))
    
    # Testing Burn In Assignments
    assert_value( data['Solver']['Default Burn In'], 5 )
    assert_value( data['Solver']['Internal']['Current Burn In'], currentBurnIn[gen] )
    
    # Testing Correctness of (min/max) Rho Update
    assert_value( data['Solver']['Min Annealing Exponent Update'], minRhoUpdate )
    assert_value( data['Solver']['Max Annealing Exponent Update'], maxRhoUpdate )
    rho =  data['Solver']['Internal']['Annealing Exponent']
    
    if (gen>0):
      assert_value(rho - prevRho >= minRhoUpdate, True)
    
    finished = data['General']['Is Finished']
    if finished == True:
        assert_value(rho - prevRho >= maxRhoUpdate-eps, False)
    if finished == False:
        assert_value(rho - prevRho <= maxRhoUpdate+eps, True)
 
    # Updates
    prevRho = rho
    gen = gen + 1
