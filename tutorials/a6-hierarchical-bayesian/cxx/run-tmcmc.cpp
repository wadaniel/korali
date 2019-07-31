#include "korali.hpp"

int main(int argc, char* argv[])
{
 auto kH = Korali::Engine();

 kH["Problem"]["Type"]  = "Hierarchical Bayesian";
 kH["Problem"]["Model"] = "Sample Psi";
 kH["Problem"]["Sub Problems"][0] = "../data/000/final.json";
 kH["Problem"]["Sub Problems"][1] = "../data/001/final.json";
 kH["Problem"]["Sub Problems"][2] = "../data/002/final.json";
 kH["Problem"]["Sub Problems"][3] = "../data/003/final.json";
 kH["Problem"]["Sub Problems"][4] = "../data/004/final.json";

 // Add probability of theta given psi, one per subproblem variable.
 kH["Problem"]["Conditional Priors"][0]["Type"] = "Gaussian";
 kH["Problem"]["Conditional Priors"][0]["Mean"] = "Psi 1";
 kH["Problem"]["Conditional Priors"][0]["Standard Deviation"] = "Psi 2";

 kH["Problem"]["Conditional Priors"][1]["Type"] = "Uniform";
 kH["Problem"]["Conditional Priors"][1]["Minimum"] = 10.0;
 kH["Problem"]["Conditional Priors"][1]["Maximum"] = "Psi 3";

 kH["Problem"]["Conditional Priors"][2]["Type"] = "Gaussian";
 kH["Problem"]["Conditional Priors"][2]["Mean"] = 2.5;
 kH["Problem"]["Conditional Priors"][2]["Standard Deviation"] = 2.0;

 kH["Problem"]["Conditional Priors"][3]["Type"] = "Uniform";
 kH["Problem"]["Conditional Priors"][3]["Minimum"] = 0.0;
 kH["Problem"]["Conditional Priors"][3]["Maximum"] = 40.0;

 kH["Variables"][0]["Name"] = "Psi 1";
 kH["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 kH["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0;
 kH["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0;

 kH["Variables"][1]["Name"] = "Psi 2";
 kH["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 kH["Variables"][1]["Prior Distribution"]["Minimum"] = +5.0;
 kH["Variables"][1]["Prior Distribution"]["Maximum"] = +10.0;

 kH["Variables"][2]["Name"] = "Psi 3";
 kH["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 kH["Variables"][2]["Prior Distribution"]["Minimum"] = 70.0;
 kH["Variables"][2]["Prior Distribution"]["Maximum"] = 90.0;

 kH["Solver"]["Type"] = "TMCMC";
 kH["Solver"]["Population Size"] = 5000;
 kH["Solver"]["Termination Criteria"]["Max Generations"] = 500;

 kH.run();
}
