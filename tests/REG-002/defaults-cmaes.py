#!/usr/bin/env python3
from math import isclose

def evaluateModel( s ):
   x = s.getVariable(0)
   r = -x*x      
   s.addResult(r)

def g1(x):
    return -3.14

import korali
k = korali.initialize()
k.setModel(evaluateModel)
k.addConstraint( g1 )

k["Problem"] = "Direct Evaluation"
k["Solver"]  = "CMAES" 

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["CMAES"]["Lower Bound"] = -10.0;
k["Variables"][0]["CMAES"]["Upper Bound"] = +10.0;

k["CMAES"]["Objective"] = "Maximize"
k["CMAES"]["Sample Count"] = 32
k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 0

k["Result Directory"] = "_defaults_cmaes"

# Running Korali
k.run()

# Testing Internals

ast = k["CMAES"]["Covariance Matrix Adaption Strength"].getValue()
assert ast == 0.1

slr = k["CMAES"]["Global Success Learning Rate"].getValue()
assert slr == 0.2

icc = k["CMAES"]["Initial Cumulative Covariance"].getValue()
assert icc == -1.0

idf = k["CMAES"]["Initial Damp Factor"].getValue()
assert idf == -1.0

isc = k["CMAES"]["Initial Sigma Cumulation Factor"].getValue()
assert isc == -1.0

chi = k["CMAES"]["Internal"]["Chi Number"].getValue()
assert chi == 0.7976190476190477

cca = k["CMAES"]["Internal"]["Covariance Matrix Adaption Factor"].getValue()
assert cca == 0.0029411764705882353

cc = k["CMAES"]["Internal"]["Cumulative Covariance"].getValue()
assert cc == 0.6681246319195033

csc = k["CMAES"]["Internal"]["Current Sample Count"].getValue()
assert csc == 32

csm = k["CMAES"]["Internal"]["Current Sample Mu"].getValue()
assert csm == 16

df = k["CMAES"]["Internal"]["Damp Factor"].getValue()
assert df == 1.6644844910916576

em = k["CMAES"]["Internal"]["Effective Mu"].getValue()
assert em == 1.9609763092840191

es = k["CMAES"]["Internal"]["Evaluation Sign"].getValue()
assert es == 1.0

sr = k["CMAES"]["Internal"]["Global Success Rate"].getValue()
print(sr)

si = k["CMAES"]["Internal"]["Sigma"].getValue()
assert si == 6.0

tr = k["CMAES"]["Internal"]["Trace"].getValue()
assert tr == 36.0

# TODO: Why are these not Internal?

#cc = k["CMAES"]["Is Sigma Bounded"].getValue()
#cc = k["CMAES"]["Max Covairance Matrix Corrections"].getValue()
#cc = k["CMAES"]["Internal"]["Mu Type"].getValue()
muv = k["CMAES"]["Mu Value"].getValue()
assert muv == 16

nvl = k["CMAES"]["Normal Vector Learning Rate"].getValue()
assert nvl == 0.029411764705882353 # TODO: is that correct?

#cc = k["CMAES"]["Internal"]["Objective"].getValue()
of = k["CMAES"]["Result Output Frequency"].getValue()
sc = k["CMAES"]["Sample Count"].getValue()
assert sc == 32.0

tsc = k["CMAES"]["Target Success Rate"].getValue()
assert tsc == 0.1818

tsc = k["CMAES"]["Terminal Output Frequency"].getValue()

# Termination Criterias
tc = k["CMAES"]["Termination Criteria"]["Max Condition Covariance Matrix"]["Value"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Max Fitness"]["Value"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Max Infeasible Resamplings"]["Value"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Max Standard Deviation"]["Value"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Min Fitness"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Min Fitness Diff Threshold"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Min Standard Deviation"].getValue()
tf = k["CMAES"]["Termination Criteria"]["Min Standard Deviation Step Factor"].getValue()

# Variables


