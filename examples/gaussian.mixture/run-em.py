#!/usr/bin/env python3

# In this example...

# Importing computational model

# Creating new experiment
import korali
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Gaussian Mixture"

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "EM"

# Configuring output settings
e["File Output"]["Frequency"] = 1
e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
