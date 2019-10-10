#!/usr/bin/env python3
import korali

reps = 10
counter = 0

def increment_counter(d):
    counter = counter + 1

k = korali.initialize()

k["Problem"]["Type"] = "Execution"
k["Problem"]["Execution Model"] = increment_counter

k["Variables"][0]["Name"] = "Tmp"
k["Variables"][0]["Loaded Values"] = [0.0]

k["Solver"]["Type"] = "Executor"
k["Solver"]["Executions Per Generation"] = 100

k["Console Output"]["Verbosity"] = "Detailed"

k.run()

assert counter == reps, "Error: Repetition / Counter mismatch ({0} vs {1})\n".format(reps, counter)
