#include "korali.h"
#include "models/rosenbrock.h"

int main(int argc, char* argv[])
{
 
 //rosenbrock
 std::function<void(Korali::ModelData&)> model = [](Korali::ModelData& d) { m_rosenbrock(d.getVariables(), d.getResults()); };
 std::function<double(double* , size_t)> fc1 = [](double* arr, size_t N) { return arr[0] + arr[1] - 2.0; }; 
 std::function<double(double* , size_t)> fc2 = [](double* arr, size_t N) { return std::pow(arr[0] - 1.0,3.0) - arr[1] + 1; }; 

 int    nParams = 2; 

 auto korali = Korali::Engine();
 korali.addConstraint(fc1);
 korali.addConstraint(fc2);

 //korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"]       = "Detailed";
 korali["Problem"]["Type"] = "Direct";

 for (int i = 0; i < nParams; i++)
 {
  korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Problem"]["Variables"][i]["Type"] = "Computational";
  korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
 }
 korali["Problem"]["Variables"][0]["Minimum"] = -1.5;
 korali["Problem"]["Variables"][0]["Maximum"] = 1.5;
 korali["Problem"]["Variables"][1]["Minimum"] = -0.5;
 korali["Problem"]["Variables"][1]["Maximum"] = 2.5;

 korali["Solver"]["Method"] = "CCMA-ES";
 korali["Solver"]["Num Samples"] = 10;
 korali["Solver"]["Num Viability Samples"] = 4;
 korali["Solver"]["Adaption Size"] = 0.1;
 korali["Solver"]["Termination Criteria"]["Max Generations"]["Value"] = 1000;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-12;
 korali["Solver"]["Termination Criteria"]["Fitness"]["Value"] = -1e-7;
 
 korali.run(model);
 
 return 0;

}
