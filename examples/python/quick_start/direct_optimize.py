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

korali["Solver"]["Method"] = "CMA-ES"
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
korali["Solver"]["Lambda"] = 5
korali["Solver"]["Objective"] = "Maximize"

korali.run(F)
