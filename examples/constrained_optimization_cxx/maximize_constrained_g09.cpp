#include "korali.h"
#include "models/g09.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();


 korali["Problem"] = "Direct Evaluation";
 korali["Solver"]  = "CCMA-ES";

 korali["Verbosity"] = "Detailed";
 
 int nParams = 7;
 for (int i = 0; i < nParams; i++){
  korali["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Variables"][i]["CCMA-ES"]["Lower Bound"] = -10.0;
  korali["Variables"][i]["CCMA-ES"]["Upper Bound"] = +10.0;
 }

 korali["CCMA-ES"]["Adaption Size"] = 0.1;
 korali["CCMA-ES"]["Sample Count"] = 8;
 korali["CCMA-ES"]["Viability Sample Count"] = 2;
 korali["CCMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = -680.630057374402 - 1e-4;
 korali["CCMA-ES"]["Termination Criteria"]["Min DeltaX"]["Value"] = 1e-12;

 korali.setModel([](Korali::ModelData& d) { g09(d.getVariables(), d.getResults()); });
 korali.addConstraint(g1);
 korali.addConstraint(g2);
 korali.addConstraint(g3);
 korali.addConstraint(g4);


 korali.run();

 return 0;

}
