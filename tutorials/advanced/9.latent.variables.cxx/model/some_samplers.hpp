#ifndef _SAMPLER_HPP_
#define _SAMPLER_HPP_
#include "korali.hpp"

#include <vector>
#include <random>
/*
void direct(korali::Sample& k)
{
  float x = k["Parameters"][0];
  k["Evaluation"] = -0.5*x*x;
}*/

void dummySampler(korali::Sample& k);

class MCMCLatentSampler {

    //  define any parameters
    std::vector<double> previousSampleMeans;
    std::vector<double> currentHyperparameters;
    //std::vector<double> previousLatentSampleMeans;
    std::vector<std::vector<double>> currentSamples;
    std::vector<std::vector<double>> previousSamples;
    int numberLatent;
    int numberHyperparameters;

    bool sample_discrete = false;
    int min_if_discrete = 0;
    int max_if_discrete = 0;

    std::function<void(korali::Sample&)> zeta_func;
    std::function<void(korali::Sample&)> S_func;
    std::function<void(korali::Sample&)> phi_func;



    std::mt19937 gen;
    std::uniform_int_distribution<int> random_int;

/*
    uint64_t zeta_func;
    uint64_t S_func;
    uint64_t phi_func;

*/
    // ... todo

    // any set up to be done?

    public:

    MCMCLatentSampler(int numberLatentVars, int numberHyperparams,
                    std::vector<double> initialLatentValues, std::vector<double> initialHyperparams,
                    std::function<void(korali::Sample&)> zeta_, std::function<void(korali::Sample&)> S_,
                            std::function<void(korali::Sample&)> phi_, bool sample_discrete_ = false,
                            int min_if_discrete_ = 0, int max_if_discrete_ = 0);
    void initialize();
    void sampleLatent(korali::Sample& k);
};

#endif
