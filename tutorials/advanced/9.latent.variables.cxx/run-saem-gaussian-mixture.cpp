#include "model/some_samplers.hpp"
#include "model/model.hpp"
#include "korali.hpp"

#include <vector>
#include <random>
#include <string>

ExampleDistribution2 distrib2 = ExampleDistribution2();

void distrib2_S(korali::Sample& s);
void distrib2_zeta(korali::Sample& s);
void distrib2_phi(korali::Sample& s);

 void distrib2_S(korali::Sample& s)
 {
   distrib2.S(s);
 };
  void distrib2_zeta(korali::Sample& s)
 {
   distrib2.zeta(s);
 };
  void distrib2_phi(korali::Sample& s)
 {
   distrib2.phi(s);
 };

  // Initialize random number generation
  // (https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution)
  std::random_device rd;
  std::mt19937 gen(rd()); // unused - todo: write loops instead of std::generate to be able to use "gen"
  std::uniform_int_distribution<> random_int(0, distrib2._p.nClusters - 1);
  std::normal_distribution<float> random_normal(0, 1); // could use gen as 'sample = random_normal(gen)'

  int numberMCMCSamples = 10;
  int d2_numberLatentVars = distrib2._p.nPoints; // one for each datapoint
  int d2_numberHyperparams = distrib2._p.nDimensions * distrib2._p.nClusters + 1;

int main(int argc, char* argv[])
{
 auto k = korali::Engine();
 auto e = korali::Experiment();

    // TODO: Need to move these out of main, and create some sort of creator function that runs the generate calls. It seems.

  std::vector<double> d2_initialLatentValues(d2_numberLatentVars, 0.0);
  std::generate(d2_initialLatentValues.begin(), d2_initialLatentValues.end(), random_int);

  std::vector<double> d2_initialHyperparams(d2_numberHyperparams, 0.0);
  std::generate(d2_initialHyperparams.begin(), d2_initialHyperparams.end(), random_normal);

  MCMCLatentSampler distrib2_sampler_obj = MCMCLatentSampler(d2_numberLatentVars, d2_numberHyperparams,
         d2_initialLatentValues, d2_initialHyperparams, &distrib2_zeta, &distrib2_S,
 		&distrib2_phi );

  void distrib2_sampler(korali::Sample& s)
  {
	  distrib2_sampler_obj.sampleLatent(s);
  }

 //auto p = heat2DInit(&argc, &argv);

 e["Problem"]["Type"] = "Bayesian/Latent";
 e["Problem"]["S Of Likelihood Model"] = &distrib2_S;
 e["Problem"]["Zeta Of Likelihood Model"] = &distrib2_zeta;
 e["Problem"]["Phi Of Likelihood Model"] = &distrib2_phi;

 e["Solver"]["Type"] = "SAEM";
 e["Solver"]["Number Markov Chain Samples"] = 100;
 e["Solver"]["Termination Criteria"]["Max Generations"] = 100;
 e["Solver"]["Latent Variable Sampler"] = &distrib2_sampler;
 // e["Solver"]["Latent Variable Sampler"] = &dummySampler;

 // * Define which hyperparameters we use (all mu, and sigma)
 int variable_counter = 0;
 for(size_t cluster_idx = 0; cluster_idx < distrib2._p.nClusters; cluster_idx++){
     for(size_t dim = 0; dim < distrib2._p.nDimensions; dim++){
         e["Variables"][variable_counter]["Name"] = "mu_"+std::to_string(cluster_idx)+"_"+std::to_string(dim);
         e["Variables"][variable_counter]["Bayesian Type"] = "Hyperparameter";
         //e["Variables"][1]["Prior Distribution"] = "Uniform 1"; // Edit: I probably dont need a prior distribution for any variable
         //e["Variables"][1]["Initial Mean"] = 0;
         //e["Variables"][1]["Initial Standard Deviation"] = 3.0;
         e["Variables"][variable_counter]["Initial Value"] = d2_initialHyperparams[variable_counter]; // Initial hyperparameter value

         variable_counter++;
     }
 }

 e["Variables"][variable_counter]["Name"] = "sigma";
 e["Variables"][variable_counter]["Bayesian Type"] = "Hyperparameter";
 //e["Variables"][variable_counter]["Prior Distribution"] = "Uniform 0"; // Edit: I probably dont need a prior distribution for any variable
 e["Variables"][variable_counter]["Initial Value"] = 5.0; // Initial hyperparameter value
 e["Variables"][variable_counter]["Upper Bound"] = 15;
 e["Variables"][variable_counter]["Lower Bound"] = -15;
 variable_counter++;

// * Latent variables
 int latent_counter = 0;
 for(size_t cluster_idx = 0; cluster_idx < distrib2._p.nPoints; cluster_idx++){
     e["Variables"][variable_counter]["Name"] = "cluster_assignment_"+std::to_string(latent_counter);
     e["Variables"][variable_counter]["Bayesian Type"] = "Latent";
     e["Variables"][variable_counter]["Initial Value"] = d2_initialLatentValues[latent_counter]; // Initial hyperparameter value

     variable_counter++;
     latent_counter++;
 }


 e["Distributions"][0]["Name"] = "Uniform 0";
 e["Distributions"][0]["Type"] = "Univariate/Uniform";
 e["Distributions"][0]["Minimum"] = 0;
 e["Distributions"][0]["Maximum"] = 5;

 e["Distributions"][1]["Name"] = "Uniform 1";
 e["Distributions"][1]["Type"] = "Univariate/Uniform";
 e["Distributions"][1]["Minimum"] = -5;
 e["Distributions"][1]["Maximum"] = 5;


 k.run(e);

 return 0;
}
