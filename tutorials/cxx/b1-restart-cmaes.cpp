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

 k["Problem"]["Type"] = "Optimization";
 k["Problem"]["Objective"] = "Maximize";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Lower Bound"] = -10.0;
 k["Variables"][0]["Upper Bound"] = +10.0;

 k["Solver"]["Type"] = "CMAES";
 k["Solver"]["Sample Count"] = 5;

 k["General"]["Max Generations"] = 500;
 k["General"]["Results Output"]["Path"] = "_b1_restart_cmaes_result";
 k["General"]["Console Output"]["Frequency"] = 10;

 k.setModel([](Korali::Model& d) { evaluateModel(d.getVariables(), d.getResults()); });
 k.run();

 printf("\n\nRestarting now:\n\n");

 // Now we loadState() to resume the same experiment from generation 10
 k.loadState("_b1_restart_cmaes_result/s00010.json");

 k.run();
}
