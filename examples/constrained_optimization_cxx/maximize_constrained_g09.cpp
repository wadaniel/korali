#include "korali.h"
#include "models/g09.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Verbosity"]       = "Detailed";
 korali["Problem"]["Type"] = "Direct";

 int nParams = 7;
 for (int i = 0; i < nParams; i++)
 {
  korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Problem"]["Variables"][i]["Type"] = "Computational";
  korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
  korali["Problem"]["Variables"][i]["Minimum"] = -10.0;
  korali["Problem"]["Variables"][i]["Maximum"] = +10.0;
 }

 korali["Solver"]["Method"] = "CCMA-ES";
 korali["Solver"]["Adaption Size"] = 0.1;
 korali["Solver"]["Num Samples"] = 8;
 korali["Solver"]["Num Viability Samples"] = 2;
 korali["Solver"]["Termination Criteria"]["Fitness"]["Value"] = -680.630057374402 - 1e-4;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-12;
 
 korali.setModel([](Korali::ModelData& d) { g09(d.getVariables(), d.getResults()); });
 korali.addConstraint(g1);
 korali.addConstraint(g2);
 korali.addConstraint(g3);
 korali.addConstraint(g4);

 korali.run();
 
 return 0;

}
