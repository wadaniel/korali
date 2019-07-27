#include "korali.hpp"

int main(int argc, char* argv[])
{
 std::vector<std::string> thetaFiles =
 {
   "results/individual/001/s00005.json",
   "results/individual/002/s00005.json",
   "results/individual/003/s00005.json",
   "results/individual/004/s00005.json",
   "results/individual/005/s00005.json"
 };

 size_t N = thetaFiles.size();
 std::vector<Korali::Engine> k;
 k.resize(N);
 for (size_t i = 0; i < N; i++)
 {
  k[i] = Korali::Engine();
  k[i].loadState(thetaFiles[i]);
 }

 auto kH = Korali::Engine();

 kH["Problem"]["Type"]  = "Hierarchical Bayesian";
 kH["Problem"]["Model"] = "Sample Psi";
 for (size_t i = 0; i < N; i++) kH["Problem"]["Sub-Problems"][i] = k[i].getResults();

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
