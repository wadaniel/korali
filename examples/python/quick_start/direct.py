#!/usr/bin/env python3
import sys
import threading
import libkorali

def F( s ):
  x = s.getParameter(0)
  r = -x*x
  s.addResult(r)

if len(sys.argv) < 2 :
  print("Usage: ./direct 1(=maximize) or 2(=sample)")
  sys.exit()


korali = libkorali.Engine( F )

korali["Problem"]["Type"] = "Direct"

korali["Problem"]["Variables"][0]["Name"] = "X"
korali["Problem"]["Variables"][0]["Minimum"] = -10.0
korali["Problem"]["Variables"][0]["Maximum"] = +10.0
korali["Problem"]["Variables"][0]["Initial Mean"] = 0.0



if sys.argv[1]=='1':
  korali["Solver"]["Method"] = "CMA-ES"
  korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
  korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
  korali["Solver"]["Lambda"] = 5
  korali["Solver"]["Objective"] = "Maximize"

elif sys.argv[1]=='2':
  korali["Solver"]["Method"] = "TMCMC"
  korali["Solver"]["Covariance Scaling"] = 0.02
  korali["Solver"]["Population Size"] = 50000
  korali["Solver"]["Coefficient of Variation"] = 0.5;
  korali["Solver"]["Burn In"] = 5

else:
  print("Usage: ./direct 1(=maximize) or 2(=sample)")
  sys.exit()

korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed"

korali.run()
