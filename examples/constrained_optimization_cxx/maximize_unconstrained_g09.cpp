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

 korali["Solver"]["Method"] = "CMA-ES";
 korali["Solver"]["Lambda"] = 8;
 //korali["Solver"]["Termination Criteria"]["Fitness"]["Value"] = -680.630057374402 - 1e-4;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-12;
 
 korali.setModel([](Korali::ModelData& d) { g09_mod(d.getVariables(), d.getResults()); });

 korali.run();
 
 return 0;

}
