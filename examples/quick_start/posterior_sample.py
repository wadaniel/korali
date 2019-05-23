#!/usr/bin/env python3
import sys
sys.path.append('./model')
from posteriorModel import *
import libkorali

korali = libkorali.Engine()
x, y = getReferenceData()

korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed";
korali["Problem"]["Type"] = "Bayesian";

korali["Problem"]["Variables"][0]["Name"] = "a";
korali["Problem"]["Variables"][0]["Type"] = "Computational";
korali["Problem"]["Variables"][0]["Distribution"] = "Uniform";
korali["Problem"]["Variables"][0]["Minimum"] = -5.0;
korali["Problem"]["Variables"][0]["Maximum"] = +5.0;

korali["Problem"]["Variables"][1]["Name"] = "b";
korali["Problem"]["Variables"][1]["Type"] = "Computational";
korali["Problem"]["Variables"][1]["Distribution"] = "Uniform";
korali["Problem"]["Variables"][1]["Minimum"] = -5.0;
korali["Problem"]["Variables"][1]["Maximum"] = +5.0;

korali["Problem"]["Variables"][2]["Name"] = "Sigma";
korali["Problem"]["Variables"][2]["Type"] = "Statistical";
korali["Problem"]["Variables"][2]["Distribution"] = "Uniform";
korali["Problem"]["Variables"][2]["Minimum"] = 0.0;
korali["Problem"]["Variables"][2]["Maximum"] = 10.0;

korali["Problem"]["Reference Data"] = y;

korali["Solver"]["Method"] = "TMCMC";
korali["Solver"]["Covariance Scaling"] = 0.02;
korali["Solver"]["Population Size"] = 5000;
korali["Solver"]["Burn In"] = 5;
korali["Solver"]["Coefficient of Variation"] = 0.5;

korali.run(lambda koraliData: evaluateModel(koraliData, x));
