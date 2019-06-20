#include "korali.h"
#include "models/g09.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Problem"]   = "Direct Evaluation";
 korali["Verbosity"] = "Detailed";

 korali["Solver"] = "CMA-ES";

 int nParams = 7;
 for (int i = 0; i < nParams; i++){
  korali["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Variables"][i]["CMA-ES"]["Lower Bound"] = -10.0;
  korali["Variables"][i]["CMA-ES"]["Upper Bound"] = +10.0;
 }
 
 korali["CMA-ES"]["Sample Count"] = 8;
 korali["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = -680.630057374402 - 1e-4;
 korali["CMA-ES"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-12;
 

 korali.setModel([](Korali::ModelData& d) { g09_mod(d.getVariables(), d.getResults()); });

 korali.run();
 
 return 0;

}
