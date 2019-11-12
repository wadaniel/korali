#!/usr/bin/env python3
import korali

def dummy(d):
    return

e = korali.newExperiment()

e["Problem"]["Type"] = "Execution/Model"
e["Problem"]["Execution Model"] = dummy

e["Variables"][0]["Name"] = "X0"
e["Variables"][0]["Loaded Values"] = [1.0, 2.0, 3.0, 4.0, 5.0]

e["Variables"][1]["Name"] = "X1"
e["Variables"][1]["Loaded Values"] = [1.0, 2.0, 3.0, 4.0, 5.0]

e["Solver"]["Type"] = "Executor"
e["Solver"]["Executions Per Generation"] = 2

e["Verbosity"] = "Detailed"

k = korali.initialize()
k.run(e)

assert e["Solver"]["Internal"]["Model Evaluation Count"] == 5
assert e["Solver"]["Internal"]["Sample Count"] == 5
assert e["Solver"]["Internal"]["Variable Count"] == 2
assert e["Internal"]["Current Generation"] == 3
