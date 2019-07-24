// In this example, we demonstrate how Korali find the variable values
// that maximize the posterior in a bayesian problem where the likelihood
// is calculated by providing reference data points and their objective values.

#include "korali.hpp"
#include "model/model.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 // Selecting problem and likelihood model
 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Additive Gaussian";
 k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData();

 // Configuring the problem's variables and their prior distributions
 k["Variables"][0]["Name"] = "a";
 k["Variables"][0]["Type"] = "Computational";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +5.0;
 k["Variables"][0]["Lower Bound"] = -5.0;
 k["Variables"][0]["Upper Bound"] = +5.0;

 k["Variables"][1]["Name"] = "b";
 k["Variables"][1]["Type"] = "Computational";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = +5.0;
 k["Variables"][1]["Lower Bound"] = -5.0;
 k["Variables"][1]["Upper Bound"] = +5.0;

 k["Variables"][2]["Name"] = "Sigma";
 k["Variables"][2]["Type"] = "Statistical";
 k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][2]["Prior Distribution"]["Maximum"] = +5.0;
 k["Variables"][2]["Lower Bound"] = 0.0;
 k["Variables"][2]["Upper Bound"] = +5.0;

 // Configuring Solver
 k["Solver"]["Type"] = "CMAES";
 k["Solver"]["Sample Count"] = 12;

 // General Settings
 k["General"]["Max Generations"] = 1000;
 k["General"]["Results Output"]["Frequency"] = 5;
 k["General"]["Console Output"]["Frequency"] = 5;

 // Setting Model
 k.setModel(model);

 // Running Korali
 k.run();
}
