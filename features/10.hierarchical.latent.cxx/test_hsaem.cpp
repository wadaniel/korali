
#include "_model/model.hpp"
#include "engine.hpp"
#include "korali.hpp"
#include "modules/module.hpp"
#include "modules/problem/bayesian/latent/hierarchicalLatentCustom/hierarchicalLatentCustom.hpp"
#include "modules/problem/problem.hpp"

#include <iostream>
#include <map>
#include <vector>

/* What to test:
    - Instatiate the hierarchicalLatentLowlevel problem and check that it returns the right
       - priors ( evaluateLogPrior(korali::Sample &sample)  )
       - conditional llhs   ( evaluateLoglikelihood(korali::Sample &sample)  )
       - total.  ( evaluateLogPosterior(korali::Sample &sample)  )

*/

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
  //  auto e_ptr = new korali::Experiment();
  //  auto e = *e_ptr;

  int nIndividuals = distrib5._p.nIndividuals;
  int nDimensions = distrib5._p.nDimensions;

  e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalLatentCustom";
  e["Problem"]["Conditional Log Likelihood Function"] = &distrib5_conditional_p; // defined in model.cpp

  e["Problem"]["Data"] = distrib5._p.data; // data: length: nIndividuals; each of its entries is a vector of different length
  e["Problem"]["Data Dimensions"] = nDimensions;
  e["Problem"]["Number Individuals"] = nIndividuals;
  e["Problem"]["Latent Space Dimensions"] = nDimensions;

  e["Solver"]["Type"] = "Executor2";
  e["Solver"]["Executions Per Generation"] = 1;
  e["Solver"]["Function To Execute"] = "Evaluate logLikelihood";
  //  e["Solver"]["Function To Execute"] = "Evaluate logPrior"  ;
  std::map<std::string, std::vector<double>> parmap1;
  std::map<std::string, std::vector<std::vector<double>>> parmap2;
  parmap1["Mean"] = std::vector<double>({0., 1.});
  // parmap1["Data Point"] = std::vector<double>({1.25, 3.});
  //parmap1["Latent Variables"] = std::vector<double>({1.0, 3.0});
  // The latents are assumed to already be in z-form, that is, log(latent)
  parmap2["Latent Variables"] = std::vector<std::vector<double>>({{0.0, 1.1}, {0.0, 1.1}, {0.0, 1.1}, {0.0, 1.1}, {0.0, 1.1}});
  parmap2["Covariance Matrix"] = std::vector<std::vector<double>>({{1.0, 0.0}, {0., 1.}});
  e["Solver"]["Vector Parameters"] = parmap1;
  e["Solver"]["Vector Of Vectors Parameters"] = parmap2;
  e["Solver"]["Vector Result Names"] = std::vector({"Log Likelihood"});
  //  e["Solver"][ "Vector Result Names"] = std::vector({"Log Prior"});
  e["Solver"]["Termination Criteria"]["Max Generations"] = 250;
  e["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 100;

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

  //  auto problem = korali::Module::getModule(e["Problem"], &e);
  //  auto highlevelProblem = dynamic_cast<korali::problem::bayesian::latent::HierarchicalLatent*>(problem);
  //  std::cout << "Data dimension: " << highlevelProblem->_dataDimensions << "\n";

  k.run(e);

  return 0;
}
