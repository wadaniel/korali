
#include "_model/model.hpp"
#include "korali.hpp"

#include <vector>

HierarchicalDistribution4 distrib4 = HierarchicalDistribution4();

void distrib4_conditional_p(korali::Sample &s);

void distrib4_conditional_p(korali::Sample &s)
{
  distrib4.conditional_p(s);
}

int main(int argc, char *argv[])
{
  auto k = korali::Engine();
  auto e = korali::Experiment();

  int nIndividuals = distrib4._p.nIndividuals;

  std::vector<std::function<void(korali::Sample & s)>> logLikelihoodFunctions(nIndividuals);
  for (size_t i = 0; i < nIndividuals; i++)
  {
    std::vector<std::vector<double>> extended_data(1);
    extended_data[0] = {distrib4._p.data[i][0], float(i)};
    logLikelihoodFunctions[i] = [distrib4, i, extended_data](korali::Sample &s) -> void {
      distrib4.conditional_p(s, extended_data);
    };
  }

  e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalCustom";
  e["Problem"]["Log Likelihood Functions"] = logLikelihoodFunctions;

  // We need to add one dimension to _p.data, because one individual in the general case could have
  // more than one data point assigned
  //  std::vector<std::vector<std::vector<double>>> dataVector(nIndividuals);
  //  for (size_t i = 0; i < nIndividuals; i++)
  //  {
  //    dataVector[i].clear();
  //    dataVector[i].push_back(distrib4._p.data[i]);
  //  }
  //  e["Problem"]["Data"] = dataVector;
  //  e["Problem"]["Data Dimensions"] = 1;
  //  e["Problem"]["Number Individuals"] = nIndividuals;
  e["Problem"]["Latent Space Dimensions"] = 1;

  e["Solver"]["Type"] = "HSAEM";
  e["Solver"]["Number Samples Per Step"] = 5; // reduce further to speed up
  e["Solver"]["Termination Criteria"]["Max Generations"] = 30;
  // Set up simulated annealing
  e["Solver"]["Use Simulated Annealing"] = true;
  e["Solver"]["Simulated Annealing Decay Factor"] = 0.8;
  e["Solver"]["Simulated Annealing Initial Variance"] = 5.;
  e["Solver"]["K1"] = 10;

  e["Distributions"][0]["Name"] = "Uniform 0";
  e["Distributions"][0]["Type"] = "Univariate/Uniform";
  e["Distributions"][0]["Minimum"] = -100;
  e["Distributions"][0]["Maximum"] = 100;

  // * Define which latent variables we use (only the means - sigma is assumed known and the same for each)
  // for (size_t i = 0; i < nIndividuals; i++){
  for (size_t i = 0; i < 1; i++)
  {
    e["Variables"][i]["Name"] = "latent mean " + std::to_string(i);
    e["Variables"][i]["Initial Value"] = -5.0;
    e["Variables"][i]["Bayesian Type"] = "Latent";
    // These are set implicitly already ('Hierarchical Latent' problem doesn't have those parameters)
    //    e["Variables"][i]["Individual Index"] = i;
    //    e["Variables"][i]["Latent Space Coordinate"] = 0;
    e["Variables"][i]["Latent Variable Distribution Type"] = "Normal";
    e["Variables"][i]["Prior Distribution"] = "Uniform 0"; // not used (?) but required
  }
  e["File Output"]["Frequency"] = 50;
  e["Console Output"]["Frequency"] = 1;
  e["Console Output"]["Verbosity"] = "Detailed";

  k.run(e);

  return 0;
}
