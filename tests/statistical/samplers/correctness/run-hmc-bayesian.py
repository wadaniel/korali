#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali

####### Bayesian Problems

##### No NUTS

e = korali.Experiment()
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = False

# Configuring problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = model

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -20.0
e["Distributions"][0]["Maximum"] = +20.0

# Defining problem's variables and prior distribution for TMCMC
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = +20.0
  
# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 100
e["Solver"]["Termination Criteria"]["Max Samples"] = 10

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Target Acceptance Rate"] = 0.71

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts"

k = korali.Engine()
k.run(e)

##### Euclidean (No Diagonal)

e = korali.Experiment()
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = False

# Configuring problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = model

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -20.0
e["Distributions"][0]["Maximum"] = +20.0

# Defining problem's variables and prior distribution for TMCMC
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = +20.0
  
# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 100
e["Solver"]["Termination Criteria"]["Max Samples"] = 10

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Version"] = 'Euclidean'
e["Solver"]["Use NUTS"] = True
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Target Acceptance Rate"] = 0.71

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts"

k = korali.Engine()
k.run(e)

##### Euclidean (Diagonal)

e = korali.Experiment()
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = False

# Configuring problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = model

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -20.0
e["Distributions"][0]["Maximum"] = +20.0

# Defining problem's variables and prior distribution for TMCMC
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = +20.0
  
# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 100
e["Solver"]["Termination Criteria"]["Max Samples"] = 10

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Version"] = 'Euclidean'
e["Solver"]["Use NUTS"] = True
e["Solver"]["Use Diagonal Metric"] = True
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Target Acceptance Rate"] = 0.71

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts"

k = korali.Engine()
k.run(e)

##### Static (No Diagonal)

e = korali.Experiment()
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = False

# Configuring problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = model

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -20.0
e["Distributions"][0]["Maximum"] = +20.0

# Defining problem's variables and prior distribution for TMCMC
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = +20.0

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 100
e["Solver"]["Termination Criteria"]["Max Samples"] = 10

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Version"] = 'Static'
e["Solver"]["Use NUTS"] = True
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Target Acceptance Rate"] = 0.71

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts"

k = korali.Engine()
k.run(e)

##### Static (Diagonal)

e = korali.Experiment()
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = False

# Configuring problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = model

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -20.0
e["Distributions"][0]["Maximum"] = +20.0

# Defining problem's variables and prior distribution for TMCMC
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = +20.0

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 100
e["Solver"]["Termination Criteria"]["Max Samples"] = 10

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Version"] = 'Static'
e["Solver"]["Use NUTS"] = True
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Use Diagonal Metric"] = True
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Target Acceptance Rate"] = 0.71

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts"

k = korali.Engine()
k.run(e)
