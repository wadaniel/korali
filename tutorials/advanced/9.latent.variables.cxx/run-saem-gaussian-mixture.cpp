#include "model/some_samplers.hpp"
#include "model/model.hpp"
#include "korali.hpp"

#include <vector>
#include <random>
#include <string>




int main(int argc, char* argv[])
{

    // TODO: Need to move these out of main, and create some sort of creator function that runs the generate calls.
    //          It seems. Or transform it into a lambda?


	// Initialize the distribution

	ExampleDistribution2  distrib2 = ExampleDistribution2();

    std::function<void(korali::Sample& s)> distrib2_S = [&distrib2](korali::Sample& s) -> void {
    	distrib2.S(s);
    } ;
    std::function<void(korali::Sample& s)> distrib2_zeta = [&distrib2](korali::Sample& s) -> void {
    	distrib2.zeta(s);
    } ;
    std::function<void(korali::Sample& s)> distrib2_phi = [&distrib2](korali::Sample& s) -> void {
    	distrib2.phi(s);
    } ;


 // Initialize random number generation
 // (https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution)
 std::random_device rd;
 std::mt19937 gen(rd());
 //std::uniform_int_distribution<int> random_int(0, distrib2._p.nClusters - 1);
 std::normal_distribution<float> random_normal(0, 1); // could use gen as 'sample = random_normal(gen)'
 std::normal_distribution<float> random_normal_2(0.5, 0.05);

 int numberMCMCSamples = 10;
 int d2_numberLatentVars = distrib2._p.nPoints; // one for each datapoint
 int d2_numberHyperparams = distrib2._p.nDimensions * distrib2._p.nClusters + 1;




  std::vector<double> d2_initialLatentValues(d2_numberLatentVars, 0.0);
  for (size_t i = 0; i < d2_numberLatentVars; i++){
      d2_initialLatentValues[i] = random_normal_2(gen);
  }
//  std::generate(d2_initialLatentValues.begin(), d2_initialLatentValues.end(), random_int);

  std::vector<double> d2_initialHyperparams(d2_numberHyperparams, 0.0);
  for (size_t i = 0; i < d2_numberHyperparams; i++){
      d2_initialHyperparams[i] = random_normal(gen);
  }
//  std::generate(d2_initialHyperparams.begin(), d2_initialHyperparams.end(), random_normal);

//  MCMCLatentSampler distrib2_sampler_obj = MCMCLatentSampler(d2_numberLatentVars, d2_numberHyperparams, // @suppress("Type cannot be resolved") // @suppress("Function cannot be resolved")
//         d2_initialLatentValues, d2_initialHyperparams, distrib2_zeta, distrib2_S, distrib2_phi,
//		 true, 0, distrib2._p.nClusters - 1 ); // @suppress("Field cannot be resolved")
  MultimodalGaussianSampler gaussian_sampler_obj = MultimodalGaussianSampler(distrib2._p.points, distrib2._p.nDimensions, distrib2._p.nClusters);

  /*void distrib2_sampler(korali::Sample& s)
  {
	  distrib2_sampler_obj.sampleLatent(s);
  }*/


//  std::function<void(korali::Sample&)> distrib2_sampler = [&distrib2_sampler_obj](korali::Sample& s) -> void {
//     distrib2_sampler_obj.sampleLatent(s);
//  };
  std::function<void(korali::Sample&)> multimodal_gaussian_sampler = [&gaussian_sampler_obj](korali::Sample& s) -> void {
	  gaussian_sampler_obj.sampleLatent(s);
  };



  auto k = korali::Engine();
  auto e = korali::Experiment();

 //auto p = heat2DInit(&argc, &argv);

 e["Problem"]["Type"] = "Bayesian/Latent";
 e["Problem"]["S Of Likelihood Model"] = distrib2_S;
 e["Problem"]["Zeta Of Likelihood Model"] = distrib2_zeta;
 e["Problem"]["Phi Of Likelihood Model"] = distrib2_phi;
 e["Problem"]["S Dimension"] = distrib2.sufficientStatisticsDimension;

 e["Solver"]["Type"] = "SAEM";
 e["Solver"]["Number Samples Per Step"] = 100;
 e["Solver"]["Termination Criteria"]["Max Generations"] = 100;
 e["Solver"]["Latent Variable Sampler"] = multimodal_gaussian_sampler;
 // e["Solver"]["Latent Variable Sampler"] = &dummySampler;


 e["Distributions"][0]["Name"] = "Uniform 0";
 e["Distributions"][0]["Type"] = "Univariate/Uniform";
 e["Distributions"][0]["Minimum"] = 0;
 e["Distributions"][0]["Maximum"] = 5;

 e["Distributions"][1]["Name"] = "Uniform 1";
 e["Distributions"][1]["Type"] = "Univariate/Uniform";
 e["Distributions"][1]["Minimum"] = -5;
 e["Distributions"][1]["Maximum"] = 5;


 e["Distributions"][2]["Name"] = "Multinomial 2";
 e["Distributions"][2]["Type"] = "Specific/Multinomial";


 // * Define which hyperparameters we use (all mu, and sigma)
 int variable_counter = 0;
 for(size_t cluster_idx = 0; cluster_idx < distrib2._p.nClusters; cluster_idx++){ // @suppress("Type cannot be resolved") // @suppress("Field cannot be resolved")
     for(size_t dim = 0; dim < distrib2._p.nDimensions; dim++){ // @suppress("Type cannot be resolved") // @suppress("Field cannot be resolved")
         e["Variables"][variable_counter]["Name"] = "mu_"+std::to_string(cluster_idx)+"_"+std::to_string(dim); // @suppress("Function cannot be resolved")
         e["Variables"][variable_counter]["Bayesian Type"] = "Hyperparameter";
         e["Variables"][variable_counter]["Prior Distribution"] = "Uniform 1"; // not used (?) but required
//         e["Variables"][variable_counter]["Initial Mean"] = 0;
//         e["Variables"][variable_counter]["Initial Standard Deviation"] = 3.0;
         e["Variables"][variable_counter]["Initial Value"] = d2_initialHyperparams[variable_counter]; // Initial hyperparameter value
         e["Variables"][variable_counter]["Upper Bound"] = e["Distributions"][1]["Maximum"]; // Initial hyperparameter value
         e["Variables"][variable_counter]["Lower Bound"] = e["Distributions"][1]["Minimum"]; // Initial hyperparameter value

         variable_counter++;
     }
 }

 e["Variables"][variable_counter]["Name"] = "sigma";
 e["Variables"][variable_counter]["Bayesian Type"] = "Hyperparameter";
 e["Variables"][variable_counter]["Prior Distribution"] = "Uniform 0"; // I probably dont need a prior distribution for any variable -- Edit: yes, seems korali demands one
 e["Variables"][variable_counter]["Initial Value"] = 2.0; // Initial hyperparameter value
 e["Variables"][variable_counter]["Upper Bound"] = 5;
 e["Variables"][variable_counter]["Lower Bound"] = 0;
 variable_counter++;

// * Latent variables
 int latent_counter = 0;
 for(size_t cluster_idx = 0; cluster_idx < distrib2._p.nPoints; cluster_idx++){ // @suppress("Type cannot be resolved") // @suppress("Field cannot be resolved")
     e["Variables"][variable_counter]["Name"] = "cluster_assignment_"+std::to_string(latent_counter); // @suppress("Function cannot be resolved")
     e["Variables"][variable_counter]["Bayesian Type"] = "Latent";
     //e["Variables"][variable_counter]["Granularity"] = 1.0; // <- does not work
     e["Variables"][variable_counter]["Prior Distribution"] = "Multinomial 2"; // not used but required
     e["Variables"][variable_counter]["Initial Value"] = d2_initialLatentValues[latent_counter]; // Initial hyperparameter value

     variable_counter++;
     latent_counter++;
 }


 e["File Output"]["Frequency"] = 50;
 e["Console Output"]["Frequency"] = 10;
 e["Console Output"]["Verbosity"] = "Detailed";

 k.run(e);

 return 0;
}
