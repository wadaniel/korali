// In this example, we demonstrate how Korali samples the posterior
// distribution in a bayesian problem where the likelihood
// is provided directly by the computational model.
// In this case, we use the MCMC method.

#include "korali.h"
#include "model/model.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 // Selecting problem and solver types.
 k["Problem"]["Type"] = "Sampling";

 // Defining problem's variables and their prior distribution
 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Initial Mean"] = 0.0;
 k["Variables"][0]["Initial Standard Deviation"] = 1.0;

 // Configuring the MCMC sampler parameters
 k["Solver"]["Type"]  = "MCMC";
 k["Solver"]["Burn In"] = 500;
 k["Solver"]["Max Chain Length"] = 5000;

 // Setting Model
 k.setModel([](Korali::Model& d) { model(d.getVariables(), d.getResults()); });

 // Running Korali
 k.run();
}
