#!/usr/bin/env python3

## In this example, we demonstrate how Korali simulates a reaction.

# Importing computational model
import sys

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Reaction"
e["Problem"]["Reactions"][0]["Equation"] = "S+I->2I"
e["Problem"]["Reactions"][0]["Rate"] = 0.0005
e["Problem"]["Reactions"][1]["Equation"] = "I->R"
e["Problem"]["Reactions"][1]["Rate"] = 0.2
e["Problem"]["Simulation Length"] = 10.


e["Variables"][0]["Name"] = "S"
e["Variables"][0]["Initial Reactant Number"] = 5000

e["Variables"][1]["Name"] = "I"
e["Variables"][1]["Initial Reactant Number"] = 5

e["Variables"][2]["Name"] = "R"
e["Variables"][2]["Initial Reactant Number"] = 0

# Configuring SSA parameters
e["Solver"]["Type"] = "SSM/SSA"
e["Solver"]["Termination Criteria"]["Max Num Simulations"] = 10
e["Solver"]["Num Bins"] = 100

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_ssa'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)
