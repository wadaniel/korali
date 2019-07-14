// In this example, we demonstrate how a Korali experiment can
// be resumed from any point (generation). This is a useful feature
// for continuing jobs after an error, or to fragment big jobs into
// smaller ones that can better fit a supercomputer queue.

#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
  // First, we run a simple Korali experiment.
  auto k = Korali::Engine();
  k.setModel([](Korali::Model& d) { evaluateModel(d.getVariables(), d.getResults()); });

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

  // Now we loadState() to resume the same experiment from generation 10
  k.loadState("_b1_restart_cmaes/s00010.json");

  k.run();

}
