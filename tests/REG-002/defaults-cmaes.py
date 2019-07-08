#!/usr/bin/env python3

# Importing computational model
import sys

def evaluateModel( s ):
   x = s.getVariable(0)
   r = -x*x      
   s.addResult(r)

# Starting Korali's Engine
import korali
k = korali.initialize()
k.setModel(evaluateModel)

# Selecting problem and solver types.
k["Problem"] = "Direct Evaluation"
k["Solver"]  = "CMAES" 

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "X";
k["Variables"][0]["CMAES"]["Lower Bound"] = -10.0;
k["Variables"][0]["CMAES"]["Upper Bound"] = +10.0;

# Configuring CMA-ES parameters
k["CMAES"]["Objective"] = "Maximize"
k["CMAES"]["Sample Count"] = 32

k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 0

# Setting output directory
k["Result Directory"] = "_defaults_cmaes"

# Running Korali
k.run()

ast = k["CMAES"]["Covariance Matrix Adaption Strength"].getValue()
print(ast)

lr = k["CMAES"]["Global Success Learning Rate"].getValue()
print(lr)

icc = k["CMAES"]["Initial Cumulative Covariance"].getValue()
print(icc)

idf = k["CMAES"]["Initial Damp Factor"].getValue()
print(idf)

isc = k["CMAES"]["Initial Sigma Cumulation Factor"].getValue()
print(isc)

chi = k["CMAES"]["Internal"]["Chi Number"].getValue()
print(chi)

cc = k["CMAES"]["Internal"]["Covariance Matrix Adaption Factor"].getValue()
cc = k["CMAES"]["Internal"]["Covariance Matrix Learning Rate"].getValue()
cc = k["CMAES"]["Internal"]["Cumulative Covariance"].getValue()
print(cc)

cc = k["CMAES"]["Internal"]["Current Sample Count"].getValue()
cc = k["CMAES"]["Internal"]["Current Sample Mu"].getValue()
cc = k["CMAES"]["Internal"]["Damp Factor"].getValue()
cc = k["CMAES"]["Internal"]["Effective Mu"].getValue()
cc = k["CMAES"]["Internal"]["Evaluation Sign"].getValue()
cc = k["CMAES"]["Internal"]["Global Success Rate"].getValue()
cc = k["CMAES"]["Internal"]["Sigma"].getValue()
cc = k["CMAES"]["Internal"]["Trace"].getValue()
#cc = k["CMAES"]["Is Sigma Bounded"].getValue()
#cc = k["CMAES"]["Internal"]["Max Covairance Matrix Corrections"].getValue()
#cc = k["CMAES"]["Internal"]["Mu Type"].getValue()
#cc = k["CMAES"]["Internal"]["Mu Value"].getValue()
cc = k["CMAES"]["Normal Vector Learning Rate"].getValue()
#cc = k["CMAES"]["Internal"]["Objective"].getValue()
#cc = k["CMAES"]["Internal"]["Sample Count"].getValue()
#cc = k["CMAES"]["Internal"]["Target Success Rate"].getValue()

# Termination Criterias

# Variables


