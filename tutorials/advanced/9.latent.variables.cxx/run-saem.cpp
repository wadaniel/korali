#include "model/some_samplers.hpp"
#include "model/model.hpp"
#include "korali.hpp"

#include <vector>

ExampleDistribution1 distrib1 = ExampleDistribution1();

void distrib1_S(korali::Sample& s);
void distrib1_zeta(korali::Sample& s);
void distrib1_phi(korali::Sample& s);

 void distrib1_S(korali::Sample& s)
 {
   distrib1.S(s);
 };
  void distrib1_zeta(korali::Sample& s)
 {
   distrib1.zeta(s);
 };
  void distrib1_phi(korali::Sample& s)
 {
   distrib1.phi(s);
 };

  int numberMCMCSamples = 10;
  int d1_numberLatentVars = 1;
  int d1_numberHyperparams = 1;
  double initialMu = 0;
  double initialSigma = 2;
  std::vector<double> d1_initialLatentValues = {initialMu};
  std::vector<double> d1_initialHyperparams = {initialSigma};

  MCMCLatentSampler distrib1_sampler_obj = MCMCLatentSampler(d1_numberLatentVars, d1_numberHyperparams,
         d1_initialLatentValues, d1_initialHyperparams, &distrib1_zeta, &distrib1_S,
 		&distrib1_phi );

  void distrib1_sampler(korali::Sample& s)
  {
	  distrib1_sampler_obj.sampleLatent(s);
  }


int main(int argc, char* argv[])
{
 auto k = korali::Engine();
 auto e = korali::Experiment();


 //auto p = heat2DInit(&argc, &argv);

 e["Problem"]["Type"] = "Bayesian/Latent";
 e["Problem"]["S Of Likelihood Model"] = &distrib1_S;
 e["Problem"]["Zeta Of Likelihood Model"] = &distrib1_zeta;
 e["Problem"]["Phi Of Likelihood Model"] = &distrib1_phi;
 e["Problem"]["S Dimension"] = 1;

 e["Solver"]["Type"] = "SAEM";
 e["Solver"]["Number Markov Chain Samples"] = 100;
 e["Solver"]["Termination Criteria"]["Max Generations"] = 100;
 e["Solver"]["Latent Variable Sampler"] = &distrib1_sampler;
 // e["Solver"]["Latent Variable Sampler"] = &dummySampler;


 e["Variables"][0]["Name"] = "sigma";
 e["Variables"][0]["Bayesian Type"] = "Hyperparameter";
 e["Variables"][0]["Prior Distribution"] = "Uniform 0"; // Edit: I probably dont need a prior distribution for any variable
 //e["Variables"][0]["Initial Mean"] = 5;
 //e["Variables"][0]["Initial Standard Deviation"] = 5.0;
 e["Variables"][0]["Initial Value"] = 5.0; // Initial hyperparameter value
 e["Variables"][0]["Upper Bound"] = 15;
 e["Variables"][0]["Lower Bound"] = -15;


 e["Variables"][1]["Name"] = "mu";
 e["Variables"][1]["Bayesian Type"] = "Latent";
 e["Variables"][1]["Prior Distribution"] = "Uniform 1"; // Edit: I probably dont need a prior distribution for any variable
 //e["Variables"][1]["Initial Mean"] = 0;
 //e["Variables"][1]["Initial Standard Deviation"] = 3.0;
 e["Variables"][1]["Initial Value"] = 0; // Initial hyperparameter value



 e["Distributions"][0]["Name"] = "Uniform 0";
 e["Distributions"][0]["Type"] = "Univariate/Uniform";
 e["Distributions"][0]["Minimum"] = 0;
 e["Distributions"][0]["Maximum"] = 5;

 e["Distributions"][1]["Name"] = "Uniform 1";
 e["Distributions"][1]["Type"] = "Univariate/Uniform";
 e["Distributions"][1]["Minimum"] = -5;
 e["Distributions"][1]["Maximum"] = 5;

/*
 e["Distributions"][2]["Name"] = "Uniform 2";
 e["Distributions"][2]["Type"] = "Univariate/Uniform";
 e["Distributions"][2]["Minimum"] = 0.6;
 e["Distributions"][2]["Maximum"] = 1.0;

 e["Distributions"][3]["Name"] = "Uniform 3";
 e["Distributions"][3]["Type"] = "Univariate/Uniform";
 e["Distributions"][3]["Minimum"] = 0.0;
 e["Distributions"][3]["Maximum"] = 20.0;

 e["Variables"][0]["Name"] = "Intensity";
 e["Variables"][0]["Bayesian Type"] = "Computational";
 e["Variables"][0]["Prior Distribution"] = "Uniform 0";
 e["Variables"][0]["Initial Mean"] = 30.0;
 e["Variables"][0]["Initial Standard Deviation"] = 5.0;

 e["Variables"][1]["Name"] = "PosX";
 e["Variables"][1]["Bayesian Type"] = "Computational";
 e["Variables"][1]["Prior Distribution"] = "Uniform 1";
 e["Variables"][1]["Initial Mean"] = 0.25;
 e["Variables"][1]["Initial Standard Deviation"] = 0.01;

 e["Variables"][2]["Name"] = "PosY";
 e["Variables"][2]["Bayesian Type"] = "Computational";
 e["Variables"][2]["Prior Distribution"] = "Uniform 2";
 e["Variables"][2]["Initial Mean"] = 0.8;
 e["Variables"][2]["Initial Standard Deviation"] = 0.1;

 e["Variables"][3]["Name"] = "Sigma";
 e["Variables"][3]["Bayesian Type"] = "Statistical";
 e["Variables"][3]["Prior Distribution"] = "Uniform 3";
 e["Variables"][3]["Initial Mean"] = 10.0;
 e["Variables"][3]["Initial Standard Deviation"] = 1.0;
 */

 k.run(e);

 return 0;
}
