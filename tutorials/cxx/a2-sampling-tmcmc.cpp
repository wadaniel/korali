// In this example, we demonstrate how Korali samples the posterior
// distribution in a bayesian problem where the likelihood
// is provided directly by the computational model.
// In this case, we use the TMCMC method.

#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 k.setLikelihood([](Korali::Model& d) { evaluateModel(d.getVariables(), d.getResults()); });

 // Selecting problem.
 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Custom";

 // Defining problem's variables and their prior distribution
 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0;

 // Configuring the TMCMC sampler parameters
 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Population Size"] = 5000;

 // General Settings
 k["General"]["Results Output"]["Path"] = "_a2_sampling_tmcmc_result";
 k["General"]["Random Seed"] = 1618;

 // Running Korali
 k.run();
}
