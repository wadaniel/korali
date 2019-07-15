// In this example, we demonstrate how Korali find the variable values
// that maximize the posterior in a bayesian problem where the likelihood
// is calculated by providing reference data points and their objective values.

#include "korali.h"
#include "model/posteriorModel.h"

int main(int argc, char* argv[])
{
  auto k = Korali::Engine();

  // Getting reference data from the model
  std::vector<double> x, y;
  getReferenceData(x, y);

  // Setting the model
  k.setModel([x](Korali::Model& d) { posteriorModel(d.getVariables(), d.getResults(), x); });

  // Selecting problem
  k["Problem"] = "Bayesian";

  // Setting up the reference likelihood for the Bayesian Problem
  k["Bayesian"]["Likelihood"]["Type"] = "Reference";
  k["Bayesian"]["Likelihood"]["Model"] = "Additive Gaussian";
  k["Bayesian"]["Likelihood"]["Reference Data"] = y;

  // Configuring the problem's variables and their prior distributions
  k["Variables"][0]["Name"] = "a";
  k["Variables"][0]["Bayesian"]["Type"] = "Computational";
  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
  k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

  k["Variables"][1]["Name"] = "b";
  k["Variables"][1]["Bayesian"]["Type"] = "Computational";
  k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
  k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
  k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

  k["Variables"][2]["Name"] = "Sigma";
  k["Variables"][2]["Bayesian"]["Type"] = "Statistical";
  k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
  k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
  k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

  // Selecting solver type
  k["Solver"] = "CMAES";

  // Configuring the CMA-ES bounds for the variables
  k["Variables"][0]["CMAES"]["Lower Bound"] = -5.0;
  k["Variables"][0]["CMAES"]["Upper Bound"] = +5.0;
  k["Variables"][1]["CMAES"]["Lower Bound"] = -5.0;
  k["Variables"][1]["CMAES"]["Upper Bound"] = +5.0;
  k["Variables"][2]["CMAES"]["Lower Bound"] = 0.0;
  k["Variables"][2]["CMAES"]["Upper Bound"] = +5.0;

  // Configuring CMA-ES parameters
  k["CMAES"]["Objective"] = "Maximize";
  k["CMAES"]["Sample Count"] = 12;

  // Reducing Output
  k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 100;\
  k["Console Output Frequency"] = 5;
  k["File Output Frequency"] = 5;

  // Setting output directory
  k["Result Directory"] = "_a3_bayesian_inference_cmaes";

  // Running Korali
  k.run();
}
