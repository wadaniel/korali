#include "korali.h"
#include "models/tr2.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Verbosity"]       = "Detailed";
 korali["Problem"]["Type"] = "Direct";

 int    nParams = 2;
 for (int i = 0; i < nParams; i++)
 {
  korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Problem"]["Variables"][i]["Type"] = "Computational";
  korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
 }

 korali["Problem"]["Variables"][0]["Minimum"] = -10.0;
 korali["Problem"]["Variables"][0]["Maximum"] = 10.0;
 korali["Problem"]["Variables"][1]["Minimum"] = -10.0;
 korali["Problem"]["Variables"][1]["Maximum"] = 10.0;

 korali["Solver"]["Method"] = "CCMA-ES";
 korali["Solver"]["Sample Count"] = 10;
 korali["Solver"]["Viability Sample Count"] = 4;
 korali["Solver"]["Adaption Size"] = 0.1;
 korali["Solver"]["Termination Criteria"]["Max Generations"]["Value"] = 1000;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-12;
 korali["Solver"]["Termination Criteria"]["Fitness"]["Value"] = -2 - 1e-8;
 
 korali.setModel([](Korali::ModelData& d) { m_tr2(d.getVariables(), d.getResults()); });
 korali.addConstraint([](double* arr, size_t N) { return 2.0 - arr[0] - arr[1]; });
 
 korali.run();

 return 0;

}
