#!/usr/bin/env python
import sys
import threading
import libkorali


def F( s, x ):
  for i in range(len(x)):
    th0 = s.getParameter(0)
    th1 = s.getParameter(1)
    r  =  th0*x[i] + th1
    s.addResult(r)



if len(sys.argv) < 2 :
  print "Usage: ./posterior 1(=maximize) or 2(=sample)"
  sys.exit()



x=[];            y=[];
x.append(1.0);   y.append(3.2069);
x.append(2.0);   y.append(4.1454);
x.append(3.0);   y.append(4.9393);
x.append(4.0);   y.append(6.0588);
x.append(5.0);   y.append(6.8425);


Fx = lambda s: F( s, x )
# Fx = lambda s,*,x=x: F( s, x )


korali = libkorali.Engine( Fx )

for i in range(len(y)):
  korali["Problem"]["Reference Data"][i] = y[i];

korali["Problem"]["Objective"] = "Posterior";


korali["Parameters"][0]["Name"] = "a";
korali["Parameters"][0]["Type"] = "Computational";
korali["Parameters"][0]["Distribution"] = "Uniform";
korali["Parameters"][0]["Minimum"] = -5.0;
korali["Parameters"][0]["Maximum"] = +5.0;
korali["Parameters"][0]["Initial Mean"] = +1.0;

korali["Parameters"][1]["Name"] = "b";
korali["Parameters"][1]["Type"] = "Computational";
korali["Parameters"][1]["Distribution"] = "Uniform";
korali["Parameters"][1]["Minimum"] = -5.0;
korali["Parameters"][1]["Maximum"] = +5.0;
korali["Parameters"][1]["Initial Mean"] = +1.0;

korali["Parameters"][2]["Name"] = "Sigma";
korali["Parameters"][2]["Type"] = "Statistical";
korali["Parameters"][2]["Distribution"] = "Uniform";
korali["Parameters"][2]["Minimum"] = 0.0;
korali["Parameters"][2]["Maximum"] = 10.0;
korali["Parameters"][2]["Initial Mean"] = +1.0;

if sys.argv[1]=='1':
  korali["Solver"]["Method"] = "CMA-ES";
  korali["Solver"]["Lambda"] = 10;
  korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-6;
  korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
  korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 3e5;

elif sys.argv[1]=='2':
  korali["Solver"]["Method"] = "TMCMC";
  korali["Solver"]["Covariance Scaling"] = 0.02;
  korali["Solver"]["Population Size"] = 5000;
  korali["Solver"]["Burn In"] = 5;
  korali["Solver"]["Coefficient of Variation"] = 0.5;
else:
  print "Usage: ./posterior 1(=maximize) or 2(=sample)"
  sys.exit()


korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed";
# korali["Live Plotting"] = true;


korali.run();
