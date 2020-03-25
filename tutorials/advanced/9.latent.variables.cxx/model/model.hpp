#ifndef _MODEL_HPP_
#define _MODEL_HPP_

#include "korali.hpp"
#include "load_data.hpp"
#include "utils.hpp"

#include <vector>
#include <cmath>
#include <random>

/*
Model 1:
A single Gaussian distribution, with
Hyperparameter = sigma, the standard deviation, and
Latent variable = mu, the mean. Assume a uniform prior distribution of the mean in [0,1].


Model 2:
Assume we have two gaussian distributions with peaks around two means, and identical
covariance = sigma.
Hyperparameters:
    mu1, mu2 and sigma
Latent variables:
    Assignment of each data point to the modes
*/


class ExponentialFamilyDistribution 
{
 public:
        pointsInfoStruct _p;
        virtual void S(korali::Sample& k) = 0;
        virtual void zeta(korali::Sample& k) = 0;
        virtual void phi(korali::Sample& k) = 0;
       // virtual void init();

       int sufficientStatisticsDimension; // Dimensionality of vectors returned by S and phi
};

class ExampleDistribution1 : public ExponentialFamilyDistribution
{
    // This might be too simple / not a good problem for E-M:
    // Take the hyperparameter as sigma of a normal distribution, and the mean mu as latent variable


    public:

         ExampleDistribution1();
        void S(korali::Sample& k) override;
        void zeta(korali::Sample& k) override;
        void phi(korali::Sample& k) override;

};

//  void distrib1_S(korali::Sample& s);
//   void distrib1_zeta(korali::Sample& s);
//   void distrib1_phi(korali::Sample& s);


    // Todo: Second test case with multiple gaussian distributions; latent variables are cluster assignments.
    // (Clustering problem)

class ExampleDistribution2 : public ExponentialFamilyDistribution
{
    // This might be too simple / not a good problem for E-M:
    // Take the hyperparameter as sigma of a normal distribution, and the mean mu as latent variable


    public:

         ExampleDistribution2();
        void S(korali::Sample& k) override;
        void zeta(korali::Sample& k) override;
        void phi(korali::Sample& k) override;


};


#endif
