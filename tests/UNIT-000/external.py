#!/usr/bin/env python3
import korali
import sys
import subprocess

def evaluate(x):
  argString = ['./ackley.py']
  for i in range(x.getVariableCount()):
    argString.append(str(x.getVariable(i)))
  retValue = subprocess.check_output(argString)
  x.addResult(float(retValue.decode()))

k = korali.Engine()
k["Verbosity"] = "Normal";

k["Problem"] = "Optimization";
for i in range(4):
  k["Variables"][i]["Name"] = "X" + str(i)

k["Conduit"]["Type"] = "External"
k["Conduit"]["Concurrent Jobs"] = int(sys.argv[1])

k["Solver"] = "CMAES"
k["CMAES"]["Sample Count"] = 10
for i in range(4):
  k["Variables"][i]["CMAES"]["Lower Bound"] = -32.0;
  k["Variables"][i]["CMAES"]["Upper Bound"] = +32.0;
k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 30

k.setModel(evaluate)

k.run()
