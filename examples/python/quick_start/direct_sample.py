#!/usr/bin/env python3
import sys
import libkorali

def F( s ):
  x = s.getParameter(0)
  r = -x*x
  s.addResult(r)

korali = libkorali.Engine()

korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed"

korali["Problem"]["Type"] = "Direct"
korali["Problem"]["Variables"][0]["Name"] = "X"
korali["Problem"]["Variables"][0]["Distribution"] = "Uniform"
korali["Problem"]["Variables"][0]["Minimum"] = -10.0
korali["Problem"]["Variables"][0]["Maximum"] = +10.0

korali["Solver"]["Method"] = "TMCMC"
korali["Solver"]["Covariance Scaling"] = 0.02
korali["Solver"]["Population Size"] = 50000
korali["Solver"]["Coefficient of Variation"] = 0.5;
korali["Solver"]["Burn In"] = 5

korali.run(F)
