#ifndef _MODEL_CPP_
#define _MODEL_CPP_

#include "korali.hpp"
#include "load_data.hpp"
#include "model.hpp"


#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <typeinfo>

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



//class ExampleDistribution1 : ExponentialFamilyDistribution
//{
    // This might be too simple / not a good problem for E-M:
    // Take the hyperparameter as sigma of a normal distribution, and the mean mu as latent variable


    ExampleDistribution1::ExampleDistribution1(){
        _p = univariateData();
    };

    void ExampleDistribution1::S(korali::Sample& k)
    {
      std::vector<double> hyperparams = k["Hyperparameters"];
      //auto latentVariables = k["Latent Variables"];
      std::vector<double> latentVariables = k["Latent Variables"];
      
      double sigma = hyperparams[0];
      double mu = latentVariables[0];

      // log(p) = -log(sigma*sqrt(pi*2)) - 0.5(x - mu)^2 * 1/sigma^2
      double sum = 0;
      double mse_per_point[_p.nPoints] = {0};
      for(size_t i = 0; i<_p.nPoints; i++){
       // double[_p.nDimensions] point = _p.points[]   // todo: = points[i*nDimensions : (i+1)*nDimensions]
          mse_per_point[i] = std::pow( _p.points[i] - mu , 2);
          sum += std::pow( _p.points[i] - mu , 2);
      }
      k["S"] = -0.5 * sum;  //or k["Evaluation"]["S"] ?


      // Later, if mu is a vector, can use the following :
      /* std::string typenm = typeid(mu).name();
      std::vector<std::string> primitive_types = {"char", "int", "float", "double"};
      std::cout << "Type: "<< typenm << std::endl;
      bool found = (std::find(primitive_types.begin(), primitive_types.end(), typenm) != primitive_types.end());
      if (!found){
        if (mu.length() != _p.nDimensions)
        korali::logError("Mean with wrong number of dimensions.\n");
      }

      // log(p) = -log(sigma*sqrt(pi*2)) - 0.5(x - mu)^2 * 1/sigma^2
      double sum = 0;
      double[_p.nPoints] mse_per_point = {0};
      for(size_t i = 0; i<_p.nPoints; i++){
       // double[_p.nDimensions] point = _p.points[]   // todo: = points[i*nDimensions : (i+1)*nDimensions]
         for (size_t j = 0; j<_p.nDimensions; i++)
            mse_per_point[i] += std::pow( _p.points[i*nDimensions + j] - mu[j] , 2);
            sum += std::pow( _p.points[i*nDimensions + j] - mu[j] , 2);
      }
      k["S"] = -0.5 * sum;  //or k["Evaluation"]["S"] ?

      */


    };

    void ExampleDistribution1::zeta(korali::Sample& k)
    {
      std::vector<double> hyperparams = k["Hyperparameters"];
      // std::vector<double> latentVars = k["Latent Variables"];

      double sigma = hyperparams[0];

      //log(sigma*sqrt(pi*2))
      k["zeta"] = std::log(sigma*std::sqrt(2*M_PI)) ;
    };
    void ExampleDistribution1::phi(korali::Sample& k)
    {
      std::vector<double> hyperparams = k["Hyperparameters"];
      // std::vector<double> latentVars = k["Latent Variables"];

      double sigma = hyperparams[0];
      // * 1/sigma^2
      k["phi"] = 1/std::pow(sigma, 2);
    };



    // Todo: Second test case with multiple gaussian distributions; latent variables are cluster assignments.
    // (Clustering problem)

#endif
