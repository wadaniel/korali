#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the TMCMC method.

# Importing computational model
import sys
sys.path.append('./model')
from directModel import *
import korali

k1 = korali.initialize()
k1.setLikelihood( evaluateModel )
k1["Problem"] = "Bayesian"
k1["Solver"] = "TMCMC"
k1["Bayesian"]["Likelihood"]["Type"] = "Direct"
k1["Variables"][0]["Name"] = "X"
k1["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k1["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
k1["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0
k1["TMCMC"]["Population Size"] = 5000
k1.run()

k2 = korali.initialize()
k2.setLikelihood( evaluateModel )
k2["Problem"] = "Bayesian"
k2["Solver"] = "TMCMC"
k2["Bayesian"]["Likelihood"]["Type"] = "Direct"
k2["Variables"][0]["Name"] = "X"
k2["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k2["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
k2["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0
k2["TMCMC"]["Population Size"] = 5000

kH = korali.initialize()
kH["Problem"] = "Hierarchical Bayesian"
kH["Hierarchical Bayesian"]["Operation Type"] = "Sample Psi"
kH.addSubProblem(k1)
kH.addSubProblem(k2)

kH["Variables"][0]["Name"] = "Psi 1"
kH["Variables"][0]["Hierarchical Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][0]["Hierarchical Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][0]["Hierarchical Bayesian"]["Prior Distribution"]["Maximum"] = +10.0

kH["Variables"][1]["Name"] = "Psi 2"
kH["Variables"][1]["Hierarchical Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][1]["Hierarchical Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][1]["Hierarchical Bayesian"]["Prior Distribution"]["Maximum"] = +10.0

kH["Solver"] = "TMCMC"
kH["TMCMC"]["Population Size"] = 5000
kH["Result Directory"] = "_c1_hierarchical_bayesian_result"

kH.run()
