#!/usr/bin/env python3
import sys
import libkorali
import argparse



def F( s ):
  x = s.getParameter(0)
  r = -x*x
  s.addResult(r)


parser = argparse.ArgumentParser()
parser.add_argument( "-s", "--solver",
                     choices = ["cmaes","tmcmc"],
                     help="Optimize or sample a direct problem")

if len(sys.argv)==1:
  parser.print_help(sys.stderr)
  sys.exit(1)

args = parser.parse_args()



korali = libkorali.Engine( F )

korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed"

korali["Problem"]["Type"] = "Direct"

korali["Problem"]["Variables"][0]["Name"] = "X"
korali["Problem"]["Variables"][0]["Distribution"] = "Uniform"
korali["Problem"]["Variables"][0]["Minimum"] = -10.0
korali["Problem"]["Variables"][0]["Maximum"] = +10.0


if args.solver == "cmaes":
  korali["Solver"]["Method"] = "CMA-ES"
  korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
  korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
  korali["Solver"]["Lambda"] = 5
  korali["Solver"]["Objective"] = "Maximize"


if args.solver == "tmcmc":
  korali["Solver"]["Method"] = "TMCMC"
  korali["Solver"]["Covariance Scaling"] = 0.02
  korali["Solver"]["Population Size"] = 50000
  korali["Solver"]["Coefficient of Variation"] = 0.5;
  korali["Solver"]["Burn In"] = 5


korali.run()
