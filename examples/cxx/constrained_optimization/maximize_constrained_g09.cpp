#include "korali.h"
#include "models/g09.h"

int main(int argc, char* argv[])
{
 
 //rosenbrock
 std::function<void(Korali::ModelData&)> model = [](Korali::ModelData& d) { g09(d.getParameters(), d.getResults()); }; 

 int nParams = 7; 
 double lower = -10;
 double upper = 10;
 double* tmp = (double*) malloc(0);
 auto korali = Korali::Engine();
 korali.addConstraint(g1);
 korali.addConstraint(g2);
 korali.addConstraint(g3);
 korali.addConstraint(g4);

 //korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"]       = "Detailed";
 korali["Conduit"]["Type"] = "Sequential";
 korali["Problem"]["Type"] = "Direct";

 for (int i = 0; i < nParams; i++)
 {
  korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Problem"]["Variables"][i]["Type"] = "Computational";
  korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
  korali["Problem"]["Variables"][i]["Minimum"] = lower;
  korali["Problem"]["Variables"][i]["Maximum"] = upper;
 }

 korali["Solver"]["Method"] = "CCMA-ES";
 korali["Solver"]["Adaption Size"] = 0.1;
 korali["Solver"]["Num Samples"] = 8;
 korali["Solver"]["Num Viability Samples"] = 2;
 korali["Solver"]["Termination Criteria"]["Fitness"] = -680.630057374402 - 1e-4;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-12;
 
 korali.run(model);
 
 return 0;

}
