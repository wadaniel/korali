
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

  e["Problem"]["Latent Space Dimensions"] = nDimensions;
  e["Problem"]["Diagonal Covariance"] = true;

  e["Solver"]["Type"] = "LatentVariableFIM";
  e["Solver"]["Number Chains"] = 1;
  e["Solver"]["MCMC Outer Steps"] = 1000;
  e["Solver"]["MCMC Target Acceptance Rate"] = 0.4;
  e["Solver"]["MCMC Subchain Steps"] = {2, 2, 0};
  e["Solver"]["Termination Criteria"]["Max Generations"] = 1;

  // Set values for the hyperparameters.
  // Insert the hyperparameter estimates from a run of HSAEM, for example:
  e["Solver"]["Hyperparameters Mean"] = {0.20, 1.35, 2.08, 2.67, 5.45, 4.25};
  // we can pass the covariance as its diagonal entries:
  e["Solver"]["Hyperparameters Diagonal Covariance"] = {1.11293, 1.24328, 1.02768, 0.871729, 2.77544, 1.90965};

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

  k["Conduit"]["Type"] = "Concurrent";
  k["Conduit"]["Concurrent Jobs"] = 4;

  k.run(e);

  return 0;
}
