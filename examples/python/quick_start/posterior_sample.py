#!/usr/bin/env python3
import sys
import libkorali

def F( s, x ):
 for i in range(len(x)):
  th0 = s.getParameter(0)
  th1 = s.getParameter(1)
  r  =  th0*x[i] + th1
  s.addResult(r)

x=[];            y=[];
x.append(1.0);   y.append(3.2069);
x.append(2.0);   y.append(4.1454);
x.append(3.0);   y.append(4.9393);
x.append(4.0);   y.append(6.0588);
x.append(5.0);   y.append(6.8425);

Fx = lambda s: F(s, x)

korali = libkorali.Engine()

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

for i in range(len(y)):
  korali["Problem"]["Reference Data"][i] = y[i];

korali["Solver"]["Method"] = "TMCMC";
korali["Solver"]["Covariance Scaling"] = 0.02;
korali["Solver"]["Population Size"] = 5000;
korali["Solver"]["Burn In"] = 5;
korali["Solver"]["Coefficient of Variation"] = 0.5;

korali.run(Fx);
