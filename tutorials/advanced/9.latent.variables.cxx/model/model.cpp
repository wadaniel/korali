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
    mu1, mu2 (vectors) and sigma (just one single 1-d variable) (in this order)
Latent variables:
    Assignment of each data point to the modes
*/



//class ExampleDistribution1 : ExponentialFamilyDistribution
//{
    // This might be too simple / not a good problem for E-M:
    // Take the hyperparameter as sigma of a normal distribution, and the mean mu as latent variable


ExampleDistribution1::ExampleDistribution1(){
        _p = univariateData();
        sufficientStatisticsDimension = 1;
};

void ExampleDistribution1::S(korali::Sample& k)
{
      //std::vector<double> hyperparams = k["Hyperparameters"];
      //auto latentVariables = k["Latent Variables"];
      std::vector<double> latentVariables = k["Latent Variables"];
      
      //double sigma = hyperparams[0];
      double mu = latentVariables[0];

      // log(p) = -log(sigma*sqrt(pi*2)) - 0.5(x - mu)^2 * 1/sigma^2
      double sum = 0;
      double mse_per_point[_p.nPoints] = {0};
      for(size_t i = 0; i<_p.nPoints; i++){
       // double[_p.nDimensions] point = _p.points[]   // todo: = points[i*nDimensions : (i+1)*nDimensions]
          mse_per_point[i] = std::pow( _p.points[i][0] - mu , 2);
          sum += std::pow( _p.points[i][0] - mu , 2);
      }
      k["S"] = std::vector<double>({-0.5 * sum});  //or k["Evaluation"]["S"] ?


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
      k["phi"] = std::vector<double>({1/std::pow(sigma, 2)});
};






    // Todo: Second test case with multiple gaussian distributions; latent variables are cluster assignments.
    // (Clustering problem)
        /*Model 2:
            Assume we have two gaussian distributions with peaks around two means, and identical
            covariance = sigma.
            Hyperparameters:
                mu1, mu2 and sigma (in this order)
            Latent variables:
                Assignment of each data point to the modes

            So,
              log(p) = sum_i [log(2*pi*sigma)*1/2] - sum_i [  |x_i|^2  - <2*mu_c(i), x_i>  +  |mu_c(i)|^2  ]/(2*sigma^2)

              ->
                 zeta(sigma, mu1, mu2) = N*log(2*pi*sigma)*1/2
                 S(x1, c(1), ... xN, c(N))
                            = sum_i [vec(-|x_i|^2, delta(ci=1), delta(ci=2), x_i * delta(ci=1), x_i * delta(ci=2) ) ]
                 phi(sigma, mu1, mu2)
                            = vec(1, -|mu1|^2, -|mu2|^2, 2 * mu1^T, 2 * mu2^T) * 1/(2*sigma)

        */


ExampleDistribution2::ExampleDistribution2(){
        _p = multivariateData();
        sufficientStatisticsDimension = 1 + _p.nClusters + _p.nClusters * _p.nDimensions;  // see below
};

void ExampleDistribution2::S(korali::Sample& k)
{
        /* For two clusters:
            S(x1, c(1), ... xN, c(N))
                            = sum_i [- vec(|x_i|^2, delta(ci=1), delta(ci=2), x_i * delta(ci=1), x_i * delta(ci=2) ) ]*/

      std::vector<double> assignments = k["Latent Variables"];
      if (assignments.size() != _p.nPoints)
          korali::logError("Latent variables should be exactly the cluster assignments, so there is one for each point in the sample.");

     // std::vector<int> assignments(latentVariables.size(), 0);
//      for (size_t i = 0; i < assignments.size(); i++){
//        double lvar =assignments[i];
//        //assignments[i] = std::lround(lvar);
//        //if (std::abs(lvar - assignments[i]) > 0.01 )
//           // korali::logError("Assigned latent variable was not an integer");
//        //	 std::cout << "Ignoring problem: Cluster-assignment latent variable was not an integer" << std::endl; // @suppress("Symbol is not resolved")
//        if (lvar < -0.49 )
//            std::cout << "Ignoring unresolvable problem: Latent variable was negative, should be a cluster assignment index"<< std::endl;
//            //korali::logError("Latent variable was negative, should be a cluster assignment index");
//        if (lvar > _p.nClusters - 0.51 )
//            std::cout << "Ignoring unresolvable problem: Latent variable was larger than highest cluster index, should be a cluster assignment index"<< std::endl;
//            //korali::logError("Latent variable was larger than highest cluster index, should be a cluster assignment index");
//      }
      int S_dim = 1 + _p.nDimensions * _p.nClusters + _p.nClusters;
      std::vector<double> S_vec(S_dim, 0.0);

      bool in_valid_range = true;
      for (double var : assignments){
        if (var <= -0.5 || var >= _p.nClusters - 0.51){
            in_valid_range = false;
            break;
        }
      }
      if (!in_valid_range){ // If it's invalid, return probability 0 -- circumvent this by casting input values to valid values before calling this function
        S_vec[0] = -korali::Inf;
        k["S"] = S_vec;
        return;
      }

      for(size_t i = 0; i<_p.nPoints; i++){ // @suppress("Field cannot be resolved")
          S_vec[0] -= l2_norm_squared(_p.points[i]); // @suppress("Field cannot be resolved")
          int cluster = std::lround(assignments[i]); // should be zero or one
          S_vec[cluster + 1] += 1;
          // to get <mu_c(i) , x_i>, add x_i to the part that will be summed with mu_c(i):
          auto mu_ci_location = &S_vec[_p.nClusters + 1 + cluster * _p.nDimensions];
          std::transform(mu_ci_location, mu_ci_location + _p.nDimensions, _p.points[i].begin(),
        		  	  	 mu_ci_location, std::plus<double>());
          //std::copy_n(_p.points[i].begin(), _p.points[i].size(), &S_vec[_p.nClusters + 1 + cluster * _p.nDimensions]);
          //S_vec[_p.nClusters + 1 + cluster * _p.nDimensions] = _p.points[i]; // c++ does not work like this
      }

      k["S"] = S_vec;

      // Later, if mu is a vector, can use the following :
      /* std::string typenm = typeid(mu).name();
      std::vector<std::string> primitive_types = {"char", "int", "float", "double"};
      std::cout << "Type: "<< typenm << std::endl;
      bool found = (std::find(primitive_types.begin(), primitive_types.end(), typenm) != primitive_types.end());
      if (!found){
        if (mu.length() != _p.nDimensions)
        korali::logError("Mean with wrong number of dimensions.\n");
      }
      */


};

void ExampleDistribution2::zeta(korali::Sample& k)
{
      std::vector<double> hyperparams = k["Hyperparameters"];
      // std::vector<double> latentVars = k["Latent Variables"];

      if (hyperparams.size() != _p.nDimensions * _p.nClusters + 1)
          korali::logError("Hyperparameters should be one mean vector per cluster, plus a 1D variable sigma. The dimension of the hyperparameter vector did not match this.");

      double sigma = hyperparams[_p.nClusters * _p.nDimensions];

      //log(sigma*sqrt(pi*2))
      k["zeta"] = _p.nPoints * std::log(sigma*std::sqrt(2*M_PI)) ; // Not sure whether std::sqrt is slower or std::pow. Todo: move sqrt out of the log?
};


void ExampleDistribution2::phi(korali::Sample& k)
{
      std::vector<double> hyperparams = k["Hyperparameters"];
      double sigma = hyperparams[_p.nClusters * _p.nDimensions];
      // std::vector<double> latentVars = k["Latent Variables"];

      if (hyperparams.size() != _p.nDimensions * _p.nClusters + 1)
          korali::logError("Hyperparameters should be one mean vector per cluster, plus a 1D variable sigma. The dimension of the hyperparameter vector did not match this.");

      std::vector<std::vector<double>> mus(0);
      for (size_t i = 0; i < _p.nClusters; i++){
        std::vector<double>::const_iterator first = hyperparams.begin() + i * _p.nDimensions;
        std::vector<double>::const_iterator last = hyperparams.begin() + (i + 1) * _p.nDimensions ;
        std::vector<double> mu(first, last);
        if(mu.size() != _p.nDimensions)
            korali::logError("Implementation error, dimensions did not match");
        mus.push_back(mu);
      }
      /* For two variables:
            phi(sigma, mu1, mu2) = vec(1, -|mu1|^2, -|mu2|^2, mu1^T, mu2^T) * 1/(2*sigma)  */
      int phi_dim = 1 + _p.nDimensions * _p.nClusters + _p.nClusters;
      std::vector<double> phi(phi_dim);
      phi[0] = 1.0;
      std::vector<double> mu_x_two(mus[0].size());
      for (size_t i = 0; i < _p.nClusters; i++){
        phi[i + 1] = - l2_norm_squared(mus[i]);
        // mu * 2:
        std::transform(mus[i].begin(), mus[i].end(), mu_x_two.begin(), [](auto& c){return c * 2.0;}); // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved")
        std::copy_n(mu_x_two.begin(), mu_x_two.size(), &phi[1 + _p.nClusters + i * _p.nDimensions]);
        //phi[1 + _p.nClusters + i * _p.nDimensions] = mus[i]; // does this work, setting a subset of a vector with a vector? --no
      }
      // *1/(2*sigma):
      std::transform(phi.begin(), phi.end(), phi.begin(), [&sigma](auto& c){return c/(2*sigma*sigma);}); // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved")
      k["phi"] = phi;
};


 /*   distrib2 = ExampleDistribution2();

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
     };*/


#endif
