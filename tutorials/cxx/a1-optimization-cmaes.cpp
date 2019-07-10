#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
 for (int i = 0; i < 11; i++)
{
 auto k = Korali::Engine();

 k.setModel([](Korali::ModelData& d) { evaluateModel(d.getVariables(), d.getResults()); });

 k["Problem"] = "Direct Evaluation";
 k["Solver"]  = "CMAES";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["CMAES"]["Lower Bound"] = -10.0;
 k["Variables"][0]["CMAES"]["Upper Bound"] = +10.0;

 k["CMAES"]["Objective"] = "Maximize";
 k["CMAES"]["Sample Count"] = 32;

 k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 500;

 k["Verbosity"] = "Silent";
 k["Result Directory"] = "_a1_optimization_cmaes_result";
 k.run();
 }
}
