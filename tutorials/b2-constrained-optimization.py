#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model, subject to a set of
## constraints.

# Importing the computational model and constraints
import sys
sys.path.append('./model')
from g09 import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem type.
k["Problem"] = "Direct Evaluation";

# Setting model and constraints
k.setModel( g09 );
k.addConstraint( g1 );
k.addConstraint( g2 );
k.addConstraint( g3 );
k.addConstraint( g4 );

# Creating 7 variables
nParams = 7;
for i in range(nParams) :
  k["Variables"][i]["Name"] = "X" + str(i);

# Selecting the CCMA-ES solver.
k["Solver"]  = "CCMA-ES";

#Setting up the variables CCMA-ES bounds
for i in range(nParams) :
  k["Variables"][i]["CCMA-ES"]["Lower Bound"] = -10.0;
  k["Variables"][i]["CCMA-ES"]["Upper Bound"] = +10.0;

# Configuring the constrained optimizer CCMA-ES
k["CCMA-ES"]["Adaption Size"] = 0.1;
k["CCMA-ES"]["Sample Count"] = 8;
k["CCMA-ES"]["Viability Sample Count"] = 2;
k["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = -680.630057374402 - 1e-4;

# Running Korali
k.run();
