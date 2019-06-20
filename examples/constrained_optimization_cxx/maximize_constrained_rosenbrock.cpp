#include "korali.h"
#include "models/rosenbrock.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Problem"] = "Direct Evaluation";
 korali["Solver"]  = "CCMA-ES";
 
 korali["Verbosity"] = "Detailed";

 int nParams = 2;
 for (int i = 0; i < nParams; i++) korali["Variables"][i]["Name"] = "X" + std::to_string(i);

 korali["Variables"][0]["CCMA-ES"]["Lower Bound"] = -1.5;
 korali["Variables"][0]["CCMA-ES"]["Upper Bound"] = +1.5;
 korali["Variables"][1]["CCMA-ES"]["Lower Bound"] = -0.5;
 korali["Variables"][1]["CCMA-ES"]["Upper Bound"] = +2.5;
 
 korali["CCMA-ES"]["Sample Count"]  = 8;
 korali["CCMA-ES"]["Viability Sample Count"] = 2;
 korali["CCMA-ES"]["Adaption Size"] = 0.1;
 
 korali["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = -680.630057374402 - 1e-4;
 korali["CCMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 1000;
 

 korali.setModel([](Korali::ModelData& d) { m_rosenbrock(d.getVariables(), d.getResults()); });
 korali.addConstraint([](double* arr, size_t N) { return arr[0] + arr[1] - 2.0; });
 korali.addConstraint([](double* arr, size_t N) { return std::pow(arr[0] - 1.0,3.0) - arr[1] + 1; });

 korali.run();
 
 return 0;

}
