
#include "_model/model.hpp"
#include "korali.hpp"

#include <vector>

HierarchicalDistribution5 distrib5 = HierarchicalDistribution5();

void distrib5_conditional_p(korali::Sample &s);
void distrib5_conditional_p(korali::Sample &s)
{
  distrib5.conditional_p(s);
}

int main(int argc, char *argv[])
{
  auto k = korali::Engine();
  auto e = korali::Experiment();

  int nIndividuals = distrib5._p.nIndividuals;
  int nDimensions = distrib5._p.nDimensions;

  std::vector<std::function<void(korali::Sample & s)>> logLikelihoodFunctions(nIndividuals);
  for (size_t i = 0; i < nIndividuals; i++)
    logLikelihoodFunctions[i] = [distrib5, i](korali::Sample &s) -> void {
      distrib5.conditional_p(s, {distrib5._p.data[i]});
    };

  e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalCustom";
  e["Problem"]["Log Likelihood Functions"] = logLikelihoodFunctions; // defined in model.cpp

  //  e["Problem"]["Data"] = distrib5._p.data; // data: length: nIndividuals; each of its entries is a vector of different length
  //  e["Problem"]["Data Dimensions"] = nDimensions;
  //  e["Problem"]["Number Individuals"] = nIndividuals;
  e["Problem"]["Latent Space Dimensions"] = nDimensions;

  e["Solver"]["Type"] = "HSAEM";
  e["Solver"]["Number Samples Per Step"] = 5; // reduce further to speed up
  e["Solver"]["Termination Criteria"]["Max Generations"] = 250;
  e["Solver"]["Delta"] = 0.4;
  // Set up simulated annealing - or disable
  e["Solver"]["Use Simulated Annealing"] = true;
  e["Solver"]["Simulated Annealing Decay Factor"] = 0.9;
  e["Solver"]["Simulated Annealing Initial Variance"] = 1.;
  // e["Solver"]["K1"] = 10;
  //  e["Solver"]["N1"] = 5;
  e["Solver"]["MCMC Outer Steps"] = 1;
  e["Solver"]["MCMC Target Acceptance Rate"] = 0.3;
  e["Solver"]["N1"] = 2;
  e["Solver"]["N2"] = 2;
  e["Solver"]["N3"] = 2;
  e["Solver"]["Use Simulated Annealing"] = true;
  e["Solver"]["Simulated Annealing Decay Factor"] = 0.95;
  e["Solver"]["Simulated Annealing Initial Variance"] = 1.;

  e["Distributions"][0]["Name"] = "Uniform 0";
  e["Distributions"][0]["Type"] = "Univariate/Uniform";
  e["Distributions"][0]["Minimum"] = -100;
  e["Distributions"][0]["Maximum"] = 100;

  e["Distributions"][1]["Name"] = "Uniform 1";
  e["Distributions"][1]["Type"] = "Univariate/Uniform";
  e["Distributions"][1]["Minimum"] = 0;
  e["Distributions"][1]["Maximum"] = 100;

  e["Distributions"][2]["Name"] = "Uniform 2";
  e["Distributions"][2]["Type"] = "Univariate/Uniform";
  e["Distributions"][2]["Minimum"] = 0.0;
  e["Distributions"][2]["Maximum"] = 1.0;

  // * Define which latent variables we use (only the means - sigma is assumed known and the same for each)
  // for (size_t i = 0; i < nIndividuals; i++){
  size_t dimCounter = 0;
  for (size_t i = 0; i < distrib5._p.dNormal; i++)
  {
    e["Variables"][dimCounter]["Name"] = "(Normal) latent mean " + std::to_string(dimCounter);
    e["Variables"][dimCounter]["Initial Value"] = -5.0;
    e["Variables"][dimCounter]["Bayesian Type"] = "Latent";
    e["Variables"][dimCounter]["Latent Variable Distribution Type"] = "Normal";
    e["Variables"][dimCounter]["Prior Distribution"] = "Uniform 0"; // not used (?) but required
    dimCounter++;
  }
  for (size_t i = 0; i < distrib5._p.dLognormal; i++)
  {
    e["Variables"][dimCounter]["Name"] = "(Log-normal) latent mean " + std::to_string(dimCounter);
    e["Variables"][dimCounter]["Initial Value"] = 5.0; // Valid range: (0, infinity)
    e["Variables"][dimCounter]["Bayesian Type"] = "Latent";
    e["Variables"][dimCounter]["Latent Variable Distribution Type"] = "Log-Normal";
    e["Variables"][dimCounter]["Prior Distribution"] = "Uniform 1"; // not used (?) but required
    dimCounter++;
  }
  for (size_t i = 0; i < distrib5._p.dLogitnormal; i++)
  {
    e["Variables"][dimCounter]["Name"] = "(Logit-normal) latent mean " + std::to_string(dimCounter);
    e["Variables"][dimCounter]["Initial Value"] = 0.5; // Valid range: [0, 1)
    e["Variables"][dimCounter]["Bayesian Type"] = "Latent";
    e["Variables"][dimCounter]["Latent Variable Distribution Type"] = "Logit-Normal";
    e["Variables"][dimCounter]["Prior Distribution"] = "Uniform 2"; // not used (?) but required
    dimCounter++;
  }
  e["File Output"]["Frequency"] = 1;
  e["Console Output"]["Frequency"] = 1;
  e["Console Output"]["Verbosity"] = "Detailed";

  k.run(e);

  return 0;
}
