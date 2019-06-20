#include "korali.h"
#include "models/tr2.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Problem"] = "Direct Evaluation";
 korali["Solver"]  = "CCMA-ES";

 korali["Verbosity"] = "Detailed";
 
 int    nParams = 2;
 for (int i = 0; i < nParams; i++)
 {
  korali["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Variables"][i]["CCMA-ES"]["Lower Bound"] = -10.0;
  korali["Variables"][i]["CCMA-ES"]["Upper Bound"] = +10.0;
 }

 korali["CCMA-ES"]["Sample Count"] = 10;
 korali["CCMA-ES"]["Viability Sample Count"] = 4;
 korali["CCMA-ES"]["Adaption Size"] = 0.1;
 korali["CCMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 1000;
 korali["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = -2 - 1e-8;
 
 korali.setModel([](Korali::ModelData& d) { m_tr2(d.getVariables(), d.getResults()); });
 korali.addConstraint([](double* arr, size_t N) { return 2.0 - arr[0] - arr[1]; });
 
 korali.run();

 return 0;

}
