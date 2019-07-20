// In this example, we demonstrate how Korali samples the posterior distribution
// in a bayesian problem where the likelihood is calculated by providing
// reference data points and their objective values.

#include "korali.h"
#include "model/model.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 // Setting up the reference likelihood for the Bayesian Problem
 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Additive Gaussian";
 k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData();

 // Configuring the problem's variables and their prior distributions
 k["Variables"][0]["Name"] = "a";
 k["Variables"][0]["Type"] = "Computational";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][1]["Name"] = "b";
 k["Variables"][1]["Type"] = "Computational";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][2]["Name"] = "Sigma";
 k["Variables"][2]["Type"] = "Statistical";
 k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][2]["Prior Distribution"]["Maximum"] = +5.0;

 // Configuring Solver
 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Population Size"] = 5000;

 // Setting the model
 k.setModel([](Korali::Model& d) { model(d.getVariables(), d.getResults(), getReferencePoints()); });

 // Running Korali
 k.run();
}
