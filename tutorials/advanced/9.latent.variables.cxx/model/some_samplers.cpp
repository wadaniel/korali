#ifndef _SAMPLER_CPP_
#define _SAMPLER_CPP_

#include "some_samplers.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <random>
/*
void direct(korali::Sample& k)
{
  float x = k["Parameters"][0];
  k["Evaluation"] = -0.5*x*x;
}*/

void dummySampler(korali::Sample& k) //, int numberLatentVars)
{

// Always samples 0

    // Test that all input parameters are set:
  // Edit: We can't have access to those: std::vector<double> latentVars = k["Latent Variables"];

  std::vector<double> hyperparams = k["Hyperparameters"];
  size_t numberSamples = k["Number Samples"];
  size_t numberLatentVars = k["Number Of Latent Variables"];

  std::vector<std::vector<double>> samples;
  for(size_t i = 0; i < numberSamples; i++){
       std::vector<double> v;
       for(size_t j=0; j < numberLatentVars; j++){
          v.push_back(0); // @suppress("Method cannot be resolved")
    }
    samples.push_back(v);
  }

  k["Samples"] = samples;
}



MCMCLatentSampler::MCMCLatentSampler(int numberLatentVars, int numberHyperparams,
                    std::vector<double> initialLatentValues, std::vector<double> initialHyperparams,
                    std::function<void(korali::Sample&)> zeta_, std::function<void(korali::Sample&)> S_,
                            std::function<void(korali::Sample&)> phi_, bool sample_discrete_ , int min_if_discrete_,
                             int max_if_discrete_)
       {

  /*  zeta_func = reinterpret_cast<uint64_t>(&zeta_);
    S_func = reinterpret_cast<uint64_t>(&S_);
    phi_func = reinterpret_cast<uint64_t>(&phi_);
*/
    zeta_func = zeta_;
    S_func = S_;
    phi_func = phi_;


    sample_discrete = sample_discrete_;
    min_if_discrete = min_if_discrete_;
    max_if_discrete = max_if_discrete_;

    if (initialLatentValues.size() != numberLatentVars ) throw std::invalid_argument("number of latent variables should match the dimension of their initial values");
    if (initialHyperparams.size() != numberHyperparams ) throw std::invalid_argument("number of hyperparameters should match the dimension of their initial values");

    previousSampleMeans = initialLatentValues;
    currentHyperparameters = initialHyperparams;

    numberLatent = numberLatentVars;
    numberHyperparameters = numberHyperparams;

    // need to always initialize those, otherwise the compiler complains
    std::random_device rd;
	gen = std::mt19937(rd());
	random_int = std::uniform_int_distribution<int>(min_if_discrete, max_if_discrete);

}


  /*  void MCMCLatentSampler::initialize(int numberSamples){

      _currentSamples.resize( _numberMarkovChainSamples);
      for (size_t j = 0; j < _numberMarkovChainSamples; j++){
          _currentSamples[j].resize( numberSamples);
          for (size_t i = 0; i < _numberLatent; i++){
              size_t idx = _latentProblem->_latentVariableIndices[i];
              _currentSamples[j][i] = _k->_variables[idx]->_initialValue;    // _k->_variables: vector of korali-variables* (std::vector<korali::Variable*>).
          }
      }
    }
*/
    void MCMCLatentSampler::sampleLatent(korali::Sample& kSample)
    {
        /*
         * probability to sample from:
         * p(d, z | theta) * p(theta) -- that's the (log-)posterior
         * - use a "Custom" bayesian problem, with our log-posterior as "Likelihood Model" , but with the current values for theta inserted.
        */

      std::vector<double> hyperparameters = kSample["Hyperparameters"];
      size_t numberSamples = kSample["Number Samples"];
      if (kSample["Number Of Latent Variables"] != numberLatent)
         korali::logError("Implementation error, number of latent variables at initialization does not fit to what was passed as variable");


    /* Create one sampling experiment to sample all latent variables. After all, the latent vars are correlated /
        have a joint distrib.
        Todo: does it make sense to re-create these experiments at every E-M step? Otherwise, how
            to automatically update the initial mean and the distribution function of the sampling experiments?*/

         auto k = korali::Engine();
         auto e = korali::Experiment();

         // Based on tutorial a2-sampling
         e["Problem"]["Type"] = "Sampling";
         e["Problem"]["Probability Function"] = [hparams=hyperparameters, this](korali::Sample& s) -> void {
                        if (! s.contains("Parameters")){
                            korali::logError("Something is trying to evaluate the likelihood without passing values for the latent variables (= parameters, here) to the sample.\n");
                        }

                        auto latent_vars = s["Parameters"].get<std::vector<double>>();

                        if (latent_vars.size() != numberLatent)
                        	korali::logError("Implementation error, latent variable vector had wrong size");

                        if (hparams.size() != numberHyperparameters)
                        	korali::logError("Implementation error, hyperparameter vector had wrong size");


                        // first, discretize the latent variables if needed
                        if (sample_discrete){
                            std::vector<int> latent_int_vars(numberLatent);
							for (size_t i = 0; i < numberLatent; i++){
								double var = latent_vars[i];
							  // assign a random cluster if out of bounds
//								if (var < min_if_discrete - 0.49 || var > max_if_discrete + 0.49){
//									  latent_int_vars[i] = random_int(gen);
								if (var < min_if_discrete - 0.49 )
									latent_int_vars[i] = min_if_discrete;
								else {if ( var > max_if_discrete + 0.49)
									 	 latent_int_vars[i] = max_if_discrete;
									 else
										latent_int_vars[i] = std::lround(var);
								}
							}
							s["Latent Variables"] = latent_int_vars;
                        } else
                        	{s["Latent Variables"] = latent_vars;}

                        s["Hyperparameters"] = hparams; // _currentHyperparameters;

                        S_func(s);
                        zeta_func(s);
                        phi_func(s);
                        //s.run(S_func);
                        // s.run(zeta_func);
                        // s.run(phi_func); // @suppress("Invalid arguments")
                          // -> Assume these set: sample["S"], sample["zeta"] and sample["phi"]
                          if (! s.contains("S")) korali::logError("The specified likelihood model did not assign the value: 'S' to the sample.\n"); // @suppress("Invalid arguments")
                          if (! s.contains("zeta")) korali::logError("The specified likelihood model did not assign the value: 'zeta' to the sample.\n");
                          if (! s.contains("phi")) korali::logError("The specified likelihood model did not assign the value: 'phi' to the sample.\n");

                          double _zetaValue = s["zeta"].get<double>();
                          std::vector<double> _sValues = s["S"].get<std::vector<double>>();
                          std::vector<double> _phiValues = s["phi"].get<std::vector<double>>();

                          s["P(x)"] = - _zetaValue + std::inner_product(std::begin(_sValues), std::end(_sValues), // @suppress("Function cannot be resolved")
                                                                                                            std::begin(_phiValues), 0.0);

                    };

        for (size_t i=0; i < numberLatent; i++){

            // Defining problem's variables
            std::string varName = "latent_" + std::to_string(i);

            e["Variables"][i]["Name"] = varName;
            e["Variables"][i]["Initial Mean"] = previousSampleMeans[i];

            e["Variables"][i]["Initial Standard Deviation"] = 2.0;
            if (sample_discrete){
            	  e["Variables"][i]["Initial Standard Deviation"] = static_cast<float>(max_if_discrete - min_if_discrete)/2.0;
            e["Variables"][i]["Initial Mean"] = previousSampleMeans[i];

 //               e["Variables"][i]["Granularity"] = 1.0; // todo: This might simply be ignored; check in the results

//                e["Distributions"][i]["Name"] = "Uniform "+std::to_string(i);
//                e["Distributions"][i]["Type"] = "Univariate/Uniform";
//                e["Distributions"][i]["Minimum"] = min_if_discrete-0.49;
//                e["Distributions"][i]["Maximum"] = max_if_discrete+0.49;
//
//                e["Variables"][i]["Prior Distribution"] = "Uniform "+std::to_string(i);

                //e["Distributions"][i]["Name"] = "Multinomial "+std::to_string(i);
                //e["Distributions"][i]["Type"] = "Specific/Multinomial";

               // e["Variables"][i]["Prior Distribution"] = "Multinomial "+std::to_string(i);
            }
        }


        // Configuring the MCMC sampler parameters
        e["Solver"]["Type"]  = "MCMC";
        e["Solver"]["Burn In"] = 500;
        e["Solver"]["Termination Criteria"]["Max Samples"] = 5000;

        // Configuring output settings
        e["File Output"]["Frequency"] = 500;
        e["Console Output"]["Frequency"] = 500;
        e["Console Output"]["Verbosity"] = "Detailed";

        // Todo: I don't think a result path is needed (and it'd need a step id in the pathname as well)
        //e["Results"]["Path"] = "setup/results_phase_1/" + "0"*(3 - str(i).length()) +  std:to_string(i);
        k.run(e);

        std::vector<std::vector<double>> db = e["Solver"]["Sample Database"].get<std::vector<std::vector<double>>>();
        printf("Database size: %lu\n", db.size());
        /*for (size_t i = 0; i < db.size(); i++)
        {
        printf("[ ");
        for (size_t j = 0; j < db[i].size(); j++)
        printf("%f, ", db[i][j]);
        printf("]\n");
        }*/
        // TODO: modify this
        std::vector<std::vector<double>>::const_iterator first = db.end() - numberSamples;
        std::vector<std::vector<double>>::const_iterator last = db.end();
        std::vector<std::vector<double>> samples(first, last);

        // modify the samples to lie in valid range, and be a discrete integer value
		if (sample_discrete) {
			for (std::vector<double> &sample : samples){ // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
				for (double &var : sample) {
					if (var < min_if_discrete - 0.49 || var > max_if_discrete + 0.49) {
						if (var < min_if_discrete - 0.49)
						    var = min_if_discrete;
						else
						    var = max_if_discrete;
					} else {
						var = std::round(var); // @suppress("Function cannot be resolved")
					}
				}
			}
		}

        kSample["Samples"] = samples;

        // set new "previous sample means"
        for(size_t i= 0; i< numberLatent; i++){ // @suppress("Type cannot be resolved")
            double sum = 0;
            for(size_t j = 0; j < numberSamples; j++) { // @suppress("Type cannot be resolved")
                sum += samples[j][i];
                }
            previousSampleMeans[i] = sum / static_cast<double>(numberSamples);
        }

      //

    }


#endif
