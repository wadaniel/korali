// In this example, we demonstrate how a Korali experiment can
// be resumed from any point (generation). This is a useful feature
// for continuing jobs after an error, or to fragment big jobs into
// smaller ones that can better fit a supercomputer queue.

#include "korali.hpp"
#include "model/model.h"

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
 k["Solver"]["Sample Count"] = 16;

 k["General"]["Max Generations"] = 500;
 k["General"]["Console Output"]["Frequency"] = 10;
 k["General"]["Results Output"]["Frequency"] = 10;

 k.setModel([](Korali::Model& d) { model(d.getVariables(), d.getResults()); });
 k.run();

 printf("\n\nRestarting now:\n\n");

 // Now we loadState() to resume the same experiment from generation 10
 k.loadState("_korali_result/s00010.json");

 k.run();
}
