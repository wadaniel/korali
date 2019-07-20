// In this example, we demonstrate how a Korali experiment can
// be resumed from any point (generation). This is a useful feature
// for continuing jobs after an error, or to fragment big jobs into
// smaller ones that can better fit a supercomputer queue.
//
// First, we run a simple Korali experiment.

#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 k.setLikelihood([](Korali::Model& d) { evaluateModel(d.getVariables(), d.getResults()); });

 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Custom";

 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0;

 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Population Size"] = 5000;

 k["General"]["Results Output"]["Path"] = "_b1_restart_tmcmc_result";

 k.run();

 printf("\n\nRestarting now:\n\n");

 // Now we loadState() to resume the same experiment from generation 5.
 k.loadState("_b1_restart_tmcmc_result/s00001.json");

 k.run();
}
