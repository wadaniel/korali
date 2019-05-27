#!/usr/bin/env python3
import sys
sys.path.append('./model')
from posteriorModel import *
import korali

k = korali.Engine()
x, y = getReferenceData()

k["Seed"] = 0xC0FFEE;
k["Verbosity"] = "Detailed";
k["Problem"]["Type"] = "Bayesian";

k["Problem"]["Variables"][0]["Name"] = "a";
k["Problem"]["Variables"][0]["Type"] = "Computational";
k["Problem"]["Variables"][0]["Distribution"] = "Uniform";
k["Problem"]["Variables"][0]["Minimum"] = -5.0;
k["Problem"]["Variables"][0]["Maximum"] = +5.0;

k["Problem"]["Variables"][1]["Name"] = "b";
k["Problem"]["Variables"][1]["Type"] = "Computational";
k["Problem"]["Variables"][1]["Distribution"] = "Uniform";
k["Problem"]["Variables"][1]["Minimum"] = -5.0;
k["Problem"]["Variables"][1]["Maximum"] = +5.0;

k["Problem"]["Variables"][2]["Name"] = "Sigma";
k["Problem"]["Variables"][2]["Type"] = "Statistical";
k["Problem"]["Variables"][2]["Distribution"] = "Uniform";
k["Problem"]["Variables"][2]["Minimum"] = 0.0;
k["Problem"]["Variables"][2]["Maximum"] = 10.0;

k["Problem"]["Reference Data"] = x;

k["Solver"]["Method"] = "CMA-ES";
k["Solver"]["Lambda"] = 10;
k["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-6;
k["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
k["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 3e5;

k.run(lambda koraliData: evaluateModel(koraliData, x));
