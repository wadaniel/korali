#!/usr/bin/env python
import sys
import threading
import libkorali

def F( s ):
  x = s.getParameter(0)
  r = -(x-2.)**2
  s.addResult(r)

if len(sys.argv) < 2 :
  print "Usage: ./direct 1(=maximize) or 2(=sample)"
  sys.exit()


korali = libkorali.Engine( F )

korali["Parameters"][0]["Name"] = "X"
korali["Parameters"][0]["Distribution"] = "Uniform"
korali["Parameters"][0]["Type"] = "Computational"
korali["Parameters"][0]["Minimum"] = -10.0
korali["Parameters"][0]["Maximum"] = +10.0

korali["Problem"]["Objective"] = "Direct Evaluation"


if sys.argv[1]=='1':
  korali["Solver"]["Method"] = "CMA-ES"
  korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
  korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
  korali["Solver"]["Lambda"] = 5

elif sys.argv[1]=='2':
  korali["Solver"]["Method"] = "TMCMC"
  korali["Solver"]["Covariance Scaling"] = 0.02
  korali["Solver"]["Population Size"] = 5000
  korali["Solver"]["Burn In"] = 5

else:
  print "Usage: ./direct 1(=maximize) or 2(=sample)"
  sys.exit()

korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed"

korali.run()
