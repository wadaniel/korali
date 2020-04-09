#ifndef _SAMPLER_CPP_
#define _SAMPLER_CPP_

#include "some_samplers.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>

void dummySampler(korali::Sample& k)
{

// Always samples 0  - unused

  std::vector<double> hyperparams = k["Hyperparameters"];
  size_t numberSamples = k["Number Samples"];
  size_t numberLatentVars = k["Number Of Latent Variables"];

  std::vector<std::vector<double>> samples;
  for(size_t i = 0; i < numberSamples; i++){
       std::vector<double> v;
       for(size_t j=0; j < numberLatentVars; j++){
          v.push_back(0);
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
         { fprintf(stderr, "[Error] Implementation error, number of latent variables at initialization does not fit to what was passed as variable"); exit(-1); }


    /* Create one sampling experiment to sample all latent variables. Why one experiment: After all, the latent vars
        are correlated / have a joint distrib. */

         auto k = korali::Engine();
         auto e = korali::Experiment();

         // Based on tutorial a2-sampling
         e["Problem"]["Type"] = "Sampling";
         e["Problem"]["Probability Function"] = [hparams=hyperparameters, this](korali::Sample& s) -> void {
                        if (! s.contains("Parameters")){
                         { fprintf(stderr, "[Error] Something is trying to evaluate the likelihood without passing values for the latent variables (= parameters, here) to the sample.\n"); exit(-1); }
                        }

                        auto latent_vars = s["Parameters"].get<std::vector<double>>();

                        if (latent_vars.size() != numberLatent)
                        { fprintf(stderr, "[Error] Implementation error, latent variable vector had wrong size"); exit(-1); }

                        if (hparams.size() != numberHyperparameters)
                        { fprintf(stderr, "[Error] Implementation error, hyperparameter vector had wrong size"); exit(-1); }

                        s["Latent Variables"] = latent_vars;

                        s["Hyperparameters"] = hparams;

                        S_func(s);
                        zeta_func(s);
                        phi_func(s);
                          // -> Assume these set: sample["S"], sample["zeta"] and sample["phi"]
                          if (! s.contains("S")) { fprintf(stderr, "[Error] The specified likelihood model did not assign the value: 'S' to the sample.\n");  exit(-1); }// @suppress("Invalid arguments")
                          if (! s.contains("zeta")) { fprintf(stderr, "[Error] The specified likelihood model did not assign the value: 'zeta' to the sample.\n"); exit(-1); }
                          if (! s.contains("phi")) { fprintf(stderr, "[Error] The specified likelihood model did not assign the value: 'phi' to the sample.\n"); exit(-1); }

                          double _zetaValue = s["zeta"].get<double>();
                          std::vector<double> _sValues = s["S"].get<std::vector<double>>();
                          std::vector<double> _phiValues = s["phi"].get<std::vector<double>>();
                          double logP_of_x = - _zetaValue + std::inner_product(std::begin(_sValues), std::end(_sValues) ,
                                  std::begin(_phiValues), 0.0);
                          s["P(x)"] = logP_of_x;

                    };

        for (size_t i=0; i < numberLatent; i++){

            // Defining problem's variables
            std::string varName = "latent_" + std::to_string(i);

            e["Variables"][i]["Name"] = varName;
            e["Variables"][i]["Initial Mean"] = previousSampleMeans[i];

            e["Variables"][i]["Initial Standard Deviation"] = 2.0;
            if (sample_discrete){
            	  e["Variables"][i]["Initial Standard Deviation"] = static_cast<float>(max_if_discrete - min_if_discrete)/10.0;
            e["Variables"][i]["Initial Mean"] = previousSampleMeans[i];

            }
        }


        // Configuring the MCMC sampler parameters
        e["Solver"]["Type"]  = "MCMC";
        e["Solver"]["Burn In"] = 500;
        e["Solver"]["Termination Criteria"]["Max Samples"] = 5000;

        // Configuring output settings
        e["File Output"]["Frequency"] = 0;
        e["Console Output"]["Frequency"] = 0;
        e["Console Output"]["Verbosity"] = "Silent";

        // Todo: I don't think a result path is needed (and it'd need a step id in the pathname as well)
        //e["Results"]["Path"] = "setup/results_phase_1/" + "0"*(3 - str(i).length()) +  std:to_string(i);
        k.run(e);

        std::vector<std::vector<double>> db = e["Solver"]["Sample Database"].get<std::vector<std::vector<double>>>();
        //printf("Database size: %lu\n", db.size());


        std::vector<std::vector<double>>::const_iterator first = db.end() - numberSamples;
        std::vector<std::vector<double>>::const_iterator last = db.end();
        std::vector<std::vector<double>> samples(first, last);
        first = db.begin();
        last = db.begin() + numberSamples;
        std::vector<std::vector<double>> initial_samples(first, last);

        kSample["Samples"] = samples;
        kSample["Initial Samples For Debugging"] = initial_samples;

        // set new "previous sample means"
        for(size_t i= 0; i< numberLatent; i++){
            double sum = 0;
            for(size_t j = 0; j < numberSamples; j++) {
                sum += samples[j][i];
                }
            previousSampleMeans[i] = sum / static_cast<double>(numberSamples);
        }

    };

//MultimodalGaussianSampler::MultimodalGaussianSampler : gen{std::random_device{}()}, uniform_distrib(0.0, 1.0) (std::vector<std::vector<double>> points_, int nDimensions_, int nClusters_){
MultimodalGaussianSampler::MultimodalGaussianSampler (std::vector<std::vector<double>> points_, int nDimensions_, int nClusters_){
	nDimensions = nDimensions_;
	nClusters = nClusters_;
	points = points_;
	nPoints = points.size();

	std::random_device rd;
	gen = std::mt19937(rd());
	uniform_distrib = std::uniform_real_distribution<>(0.0, 1.0);

};

void MultimodalGaussianSampler::sampleLatent(korali::Sample& k){
	/* Sample assignments for each point separately, proportional to the resulting data likelihood, p(point, assignment | hyperparams) */

    std::vector<double> hyperparameters = k["Hyperparameters"];
    int nSamples = k["Number Samples"];
    if (k["Number Of Latent Variables"] != nPoints)
       { fprintf(stderr, "[Error] Number of latent variables at initialization does not fit to what was passed as variable.\n"); exit(-1); }

    // get sigma
    double sigma = hyperparameters[nClusters * nDimensions];

    // get vector of means
    std::vector<std::vector<double>> mus(nClusters);
    for (size_t i = 0; i < nClusters; i++){
      std::vector<double>::const_iterator first = hyperparameters.begin() + i * nDimensions;
      std::vector<double>::const_iterator last = hyperparameters.begin() + (i + 1) * nDimensions ;
      std::vector<double> mu(first, last);
      if(mu.size() != nDimensions)
      { fprintf(stderr, "[Error] Dimensions do not match.\n"); exit(-1); }
      mus[i] = mu;
    }

    // we sample in another order, will later transpose
    std::vector<std::vector<int>> samples_transpose(nPoints);
    for (size_t i = 0; i < nPoints; i++){

    	// First, get probabilities for each cluster
    	std::vector<double> probabilities(nClusters);
    	for (size_t j = 0; j < nClusters; j++){
    		std::vector<double> mu = mus[j];
    		double p = univariate_gaussian_probability(mu, sigma, points[i]);
    		probabilities[j] = p;
    	}

    	// normalize probabilities to one
    	probabilities = normalize(probabilities);

    	std::vector<int> pointwise_samples(nSamples);
    	for (size_t j = 0; j< nSamples; j++){
    		// sample
    		double u = uniform_distrib(gen);
    		double sum = 0.0;
    		for (size_t k = 0; k < nClusters; k++){
    			sum = sum + probabilities[k];
    			if (u < sum){
    				// cluster k has been chosen!
    				pointwise_samples[j] = k;
    				break;
    			}
    		}

    	}
    	samples_transpose[i] = pointwise_samples;
    }
    std::vector<std::vector<int>> samples = transpose(samples_transpose);

    k["Samples"] = samples;

};


#endif
