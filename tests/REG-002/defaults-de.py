#!/usr/bin/env python3
import sys
def evaluateModel( s ):
   x = s.getVariable(0)
   r = -x*x      
   s.addResult(r)

import korali
k = korali.initialize()

k.setModel(evaluateModel)

k["Problem"] = "Direct Evaluation"
k["Solver"]  = "DE" 

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["DE"]["Lower Bound"] = -10.0;
k["Variables"][0]["DE"]["Upper Bound"] = +10.0;

k["DE"]["Objective"] = "Maximize"
k["DE"]["Sample Count"] = 32
k["DE"]["Termination Criteria"]["Max Generations"]["Value"] = 0

k["Result Directory"] = "_defaults_de"

# Running Korali
k.run()
