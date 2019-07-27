#include "korali.hpp"

int main(int argc, char* argv[])
{
 auto kH = Korali::Engine();

 kH["Problem"]["Type"]  = "Hierarchical Bayesian";
 kH["Problem"]["Model"] = "Sample Psi";
 kH["Problem"]["Sub-Problems"][0] = Korali::Engine::getResults("../data/000/final.json");
 kH["Problem"]["Sub-Problems"][1] = Korali::Engine::getResults("../data/001/final.json");
 kH["Problem"]["Sub-Problems"][2] = Korali::Engine::getResults("../data/002/final.json");
 kH["Problem"]["Sub-Problems"][3] = Korali::Engine::getResults("../data/003/final.json");
 kH["Problem"]["Sub-Problems"][4] = Korali::Engine::getResults("../data/004/final.json");

 kH["Problem"]["Conditional Priors"][0]["Type"] = "Gaussian";
 kH["Problem"]["Conditional Priors"][0]["Mean"] = "Psi 1";
 kH["Problem"]["Conditional Priors"][0]["Standard Deviation"] = "Psi 2";

 kH["Problem"]["Conditional Priors"][1]["Type"] = "Uniform";
 kH["Problem"]["Conditional Priors"][1]["Minimum"] = 1.0;
 kH["Problem"]["Conditional Priors"][1]["Maximum"] = "Psi 3";

 kH["Variables"][0]["Name"] = "Psi 1";
 kH["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 kH["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0;
 kH["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0;

 kH["Variables"][1]["Name"] = "Psi 2";
 kH["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 kH["Variables"][1]["Prior Distribution"]["Minimum"] =   0.0;
 kH["Variables"][1]["Prior Distribution"]["Maximum"] = +10.0;

 kH["Variables"][2]["Name"] = "Psi 3";
 kH["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 kH["Variables"][2]["Prior Distribution"]["Minimum"] = -10.0;
 kH["Variables"][2]["Prior Distribution"]["Maximum"] = +10.0;

 kH["Solver"]["Type"] = "TMCMC";
 kH["Solver"]["Population Size"] = 5000;
 kH["General"]["Max Generations"] = 100;

 kH.run();
}
