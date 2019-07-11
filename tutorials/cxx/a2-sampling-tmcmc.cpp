// In this example, we demonstrate how Korali samples the posterior
// distribution in a bayesian problem where the likelihood
// is provided directly by the computational model.
// In this case, we use the TMCMC method.

#include "korali.h"
#include "model/evaluateModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();
 k.setLikelihood([](Korali::ModelData& d) { evaluateModel(d.getVariables(), d.getResults()); });

 // Selecting problem and solver types.
 k["Problem"] = "Bayesian";
 k["Solver"] = "TMCMC";

 // Configuring Likelihood type for the bayesian problem
 k["Bayesian"]["Likelihood"]["Type"] = "Direct";

 // Defining problem's variables and their prior distribution
 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0;

 // Configuring the TMCMC sampler parameters
 k["TMCMC"]["Population Size"] = 5000;

 // Setting output directory
 k["Result Directory"] = "_a2_sampling_tmcmc";

 // Running Korali
 k.run();
}
