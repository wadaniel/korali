#!/usr/bin/env python3

## In this example, we demonstrate how Korali samples the posterior distribution
## in a bayesian problem where the likelihood is calculated by providing
## reference data points and their objective values.

# Importing the computational model
import sys
sys.path.append('./model')
from posteriorModel import *
import korali

k = korali.Engine()
x, y = getReferenceData()
k.setModel(lambda koraliData: evaluateModel(koraliData, x))

# Selecting problem.
k["Problem"] = "Bayesian";

k["Bayesian"]["Likelihood"]["Type"] = "Reference"
k["Bayesian"]["Likelihood"]["Reference Data"] = y;

k["Variables"][0]["Name"] = "a";
k["Variables"][0]["Bayesian"]["Type"] = "Computational";
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

k["Variables"][1]["Name"] = "b";
k["Variables"][1]["Bayesian"]["Type"] = "Computational";
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

k["Variables"][2]["Name"] = "Sigma";
k["Variables"][2]["Bayesian"]["Type"] = "Statistical";
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] =   0.0;
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

# Selecting solver type.
k["Solver"] = "TMCMC"

# Configuring TMCMC parameters
k["TMCMC"]["Covariance Scaling"] = 0.02;
k["TMCMC"]["Population Size"] = 5000;
k["TMCMC"]["Min Rho Update"] = 0.0;
k["TMCMC"]["Coefficient of Variation"] = 0.5;
k["TMCMC"]["Burn In"] = 5

# Running Korali
k.run()
