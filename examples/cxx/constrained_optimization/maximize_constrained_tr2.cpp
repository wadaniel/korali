#include "korali.h"
#include "models/tr2.h"

int main(int argc, char* argv[])
{
 
 //tr2
 std::function<void(Korali::ModelData&)> model = [](Korali::ModelData& d) { m_tr2(d.getParameters(), d.getResults()); }; 
 std::function<double(double* , size_t)> fc1 = [](double* arr, size_t N) { return 2.0 - arr[0] - arr[1]; }; 

 int    nParams = 2; 

 double* tmp = (double*) malloc(0);
 auto korali = Korali::Engine();
 korali.addConstraint(fc1);

 //korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"]       = "Detailed";
 korali["Conduit"]["Type"] = "Sequential";
 korali["Problem"]["Type"] = "Direct";

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
 korali["Solver"]["Num Samples"] = 10;
 korali["Solver"]["Num Viability Samples"] = 4;
 korali["Solver"]["Adaption Size"] = 0.1;
 korali["Solver"]["Termination Criteria"]["Max Generations"] = 1000;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-12;
 korali["Solver"]["Termination Criteria"]["Fitness"] = -2 - 1e-8;
 
 korali.run(model);
 
 return 0;

}
