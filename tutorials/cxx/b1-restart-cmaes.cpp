#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 k.setModel([](Korali::ModelData& d) { evaluateModel(d.getVariables(), d.getResults()); });
 
 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "CMAES";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["CMAES"]["Lower Bound"] = -10.0;
 k["Variables"][0]["CMAES"]["Upper Bound"] = +10.0;

 k["CMAES"]["Objective"] = "Maximize";
 k["CMAES"]["Sample Count"] = 5;

 k["Result Directory"] = "_b1_restart_cmaes";

 k["Termination Criteria"]["Max Generations"] = 500;
 k["Console Output Frequency"] = 10;

 k.run();

 printf("\n\nRestarting now:\n\n");

 k.loadState("_b1_restart_cmaes/s00010.json");
 k.run();

}
