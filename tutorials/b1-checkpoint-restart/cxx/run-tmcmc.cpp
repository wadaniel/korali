// In this example, we demonstrate how a Korali experiment can
// be resumed from any point (generation). This is a useful feature
// for continuing jobs after an error, or to fragment big jobs into
// smaller ones that can better fit a supercomputer queue.

// First, we run a simple Korali experiment.

#include "korali.hpp"
#include "model/model.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"]["Type"] = "Sampling";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0;

 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Population Size"] = 5000;

 k.setModel([](Korali::Model& d) { model(d.getVariables(), d.getResults()); });
 k.run();

 printf("\n\nRestarting now:\n\n");

 // Now we loadState() to resume the same experiment from generation 5.
 k.loadState("_korali_result/s00002.json");

 k.run();
}
