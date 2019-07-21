// In this example, we demonstrate how Korali samples the posterior
// distribution in a bayesian problem where the likelihood
// is provided directly by the computational model.
// In this case, we use the TMCMC method.

#include "korali.hpp"
#include "model/model.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 // Selecting problem.
 k["Problem"]["Type"] = "Sampling";

 // Defining problem's variables and their prior distribution
 k["Variables"][0]["Name"] = "X";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0;

 // Configuring the TMCMC sampler parameters
 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Population Size"] = 5000;
 k["Solver"]["Covariance Scaling"] = 0.001;

 k.setModel([](Korali::Model& d) { model(d.getVariables(), d.getResults()); });

 // Running Korali
 k.run();
}
