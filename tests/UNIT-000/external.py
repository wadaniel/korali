#!/usr/bin/env python3
import korali
import sys
sys.path.append("model")
from runModel import *

k = korali.Engine()

 if (argc != 2) { printf("Error: this example requires 'Concurrent Jobs' passed as argument.\n"); exit(-1); }
 
k["Problem"]["Type"] = "Optimization";
k["Problem"]["Objective"] = "Maximize"
k["Problem"]["Objective Function"] = runModel

k["Problem"]["Variables"][0]["Name"] = "X0"
k["Problem"]["Variables"][0]["Lower Bound"] = -32.0;
k["Problem"]["Variables"][0]["Upper Bound"] = +32.0;

k["Problem"]["Variables"][1]["Name"] = "X1"
k["Problem"]["Variables"][1]["Lower Bound"] = -32.0;
k["Problem"]["Variables"][1]["Upper Bound"] = +32.0;

k["Problem"]["Variables"][2]["Name"] = "X2"
k["Problem"]["Variables"][2]["Lower Bound"] = -32.0;
k["Problem"]["Variables"][2]["Upper Bound"] = +32.0;

k["Problem"]["Variables"][3]["Name"] = "X3"
k["Problem"]["Variables"][3]["Lower Bound"] = -32.0;
k["Problem"]["Variables"][3]["Upper Bound"] = +32.0;
  
k["Solver"]["Type"] = "CMAES"
k["Solver"]["Population Size"] = 10
k["Solver"]["Termination Criteria"]["Max Generations"] = 10

k["Conduit"]["Type"] = "External"
k["Conduit"]["Concurrent Jobs"] = int(sys.argv[1])
  
k.run()
