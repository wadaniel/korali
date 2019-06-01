#include "korali.h"
#include "models/g09.h"

int main(int argc, char* argv[])
{
 
 //rosenbrock
 std::function<void(Korali::ModelData&)> model = [](Korali::ModelData& d) { g09_mod(d.getVariables(), d.getResults()); };

 int nParams = 7; 
 double lower = -10;
 double upper = 10;
 auto korali = Korali::Engine();

 //korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"]       = "Detailed";
 korali["Problem"]["Type"] = "Direct";

 for (int i = 0; i < nParams; i++)
 {
  korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Problem"]["Variables"][i]["Type"] = "Computational";
  korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
  korali["Problem"]["Variables"][i]["Minimum"] = lower;
  korali["Problem"]["Variables"][i]["Maximum"] = upper;
 }

 korali["Solver"]["Method"] = "CMA-ES";
 korali["Solver"]["Lambda"] = 8;
 //korali["Solver"]["Termination Criteria"]["Fitness"]["Value"] = -680.630057374402 - 1e-4;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-12;
 
 korali.run(model);
 
 return 0;

}
