#include "korali.h"
#include "model/g09.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "CMAES";
 k["Verbosity"] = "Detailed";

 k["CMAES"]["Objective"] = "Maximize";
 
 k.setModel([](Korali::ModelData& d) { g09(d.getVariables(), d.getResults()); });
 k.addConstraint(g1);
 k.addConstraint(g2);
 k.addConstraint(g3);
 k.addConstraint(g4);

 int nParams = 7;
 for (int i = 0; i < nParams; i++)
 {
  k["Variables"][i]["Name"] = "X" + std::to_string(i);
  k["Variables"][i]["CMAES"]["Lower Bound"] = -10.0;
  k["Variables"][i]["CMAES"]["Upper Bound"] = +10.0;
 }

 k["CMAES"]["Is Sigma Bounded"] = true;
 k["CMAES"]["Sample Count"] = 32;
 k["CMAES"]["Viability Sample Count"] = 4;
 
 k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 500;
 
 k["CMAES"]["Termination Criteria"]["Max Fitness"]["Enabled"] = true;
 k["CMAES"]["Termination Criteria"]["Max Fitness"]["Value"] = -680.630057374402 - 1e-4;
 
 k["Result Directory"] = "_b2_constrained_optimization";

 k.run();
}
