#ifndef _MODEL_CPP_
#define _MODEL_CPP_

#include "model.hpp"
#include "korali.hpp"
#include "load_data.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <typeinfo>
#include <vector>

/*
Model 1:
A single Gaussian distribution, with
Hyperparameter = sigma, the standard deviation, and
Latent variable = mu, the mean. Assume a uniform prior distribution of the mean in [0,1].
Assume the probability of mu is supported in [-20, 20]^nDimensions.



Model 2:
Assume we have two gaussian distributions with peaks around two means, and identical
covariance = sigma.
Hyperparameters:
    mu1, mu2 (vectors) and sigma (just one single 1-d variable) (in this order)
Latent variables:
    Assignment of each data point to the modes
*/

ExampleDistribution1::ExampleDistribution1()
{
  // * Take as hyperparameter the sigma of a normal distribution, and the mean mu as latent variable

  _p = univariateData();
  sufficientStatisticsDimension = 1;
  mu_lower_bound = -20.;
  mu_upper_bound = 20.;
  assert(mu_upper_bound > mu_lower_bound);
  mu_range = mu_upper_bound - mu_lower_bound;
};

void ExampleDistribution1::S(korali::Sample &k)
{
  std::vector<double> latentVariables = k["Latent Variables"];

  auto mu_vector = latentVariables;
  bool in_valid_range = true;
  for (double mu_coord : mu_vector)
  {
    if ((mu_coord < mu_lower_bound) || (mu_coord > mu_upper_bound))
    {
      in_valid_range = false;
      k["S"] = std::vector<double>({-korali::Inf}); // => probability is zero
      return;
    }
  }

  // log(p) = -log(sigma*sqrt(pi*2)) - 0.5(x - mu)^2 * 1/sigma^2
  double sum = 0;
  double mse;
  std::vector<double> vector_diff(_p.nDimensions, 0.0);
  for (size_t i = 0; i < _p.nPoints; i++)
  {
    std::transform(_p.points[i].begin(), _p.points[i].end(), mu_vector.begin(), vector_diff.begin(), std::minus<double>());
    mse = l2_norm_squared(vector_diff);
    sum += mse;
  }
  k["S"] = std::vector<double>({-sum});
};

void ExampleDistribution1::zeta(korali::Sample &k)
{
  std::vector<double> hyperparams = k["Hyperparameters"];

  double sigma = hyperparams[0];
  double log_hypercube_volume = _p.nPoints * std::log(mu_range);
  // \__ to get a normalized distribution in both x and mu, need to multiply P by a uniform distribution in mu

  //log(sigma*sqrt(pi*2))
  k["zeta"] = std::log(sigma * std::sqrt(2 * M_PI)) * _p.nPoints + log_hypercube_volume;
};

void ExampleDistribution1::phi(korali::Sample &k)
{
  std::vector<double> hyperparams = k["Hyperparameters"];
  // std::vector<double> latentVars = k["Latent Variables"];

  double sigma = hyperparams[0];
  // * 1/(2 * sigma^2)
  k["phi"] = std::vector<double>({0.5 / std::pow(sigma, 2)});
};

/* @brief Model 2:
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
ExampleDistribution2::ExampleDistribution2()
{
  _p = multivariateData();
  sufficientStatisticsDimension = 1 + _p.nClusters + _p.nClusters * _p.nDimensions; // see below
};

/* @brief   For two clusters:
            S(x1, c(1), ... xN, c(N))
                            = sum_i [- vec(|x_i|^2, delta(ci=1), delta(ci=2), x_i * delta(ci=1), x_i * delta(ci=2) ) ]*/
void ExampleDistribution2::S(korali::Sample &k)
{
  std::vector<double> assignments = k["Latent Variables"];
  if (assignments.size() != _p.nPoints)
  {
    fprintf(stderr, "[Error] Latent variables should be exactly the cluster assignments, so there is one for each point in the sample.");
    exit(-1);
  }

  int S_dim = 1 + _p.nDimensions * _p.nClusters + _p.nClusters;
  std::vector<double> S_vec(S_dim, 0.0);

  bool in_valid_range = true;
  for (double var : assignments)
  {
    if (var <= -0.5 || var >= _p.nClusters - 0.51)
    {
      in_valid_range = false;
      break;
    }
  }
  if (!in_valid_range)
  { // If it's invalid, return probability 0
    S_vec[0] = -korali::Inf;
    k["S"] = S_vec;
    return;
  }

  for (size_t i = 0; i < _p.nPoints; i++)
  {
    S_vec[0] -= l2_norm_squared(_p.points[i]);
    int cluster = std::lround(assignments[i]); // should be zero or one
    S_vec[cluster + 1] += 1;
    // to get <mu_c(i) , x_i>, add x_i to the part that will be summed with mu_c(i):
    auto mu_ci_location = &S_vec[_p.nClusters + 1 + cluster * _p.nDimensions];
    std::transform(mu_ci_location, mu_ci_location + _p.nDimensions, _p.points[i].begin(), mu_ci_location, std::plus<double>());
  }
  k["S"] = S_vec;
};

void ExampleDistribution2::zeta(korali::Sample &k)
{
  std::vector<double> hyperparams = k["Hyperparameters"];

  if (hyperparams.size() != _p.nDimensions * _p.nClusters + 1)
  {
    fprintf(stderr, "[Error] Hyperparameters should be one mean vector per cluster, plus a 1D variable sigma. The dimension of the hyperparameter vector did not match this.");
    exit(-1);
  }

  double sigma = hyperparams[_p.nClusters * _p.nDimensions];

  //log(sigma*sqrt(pi*2))
  k["zeta"] = _p.nPoints * std::log(sigma * std::sqrt(2 * M_PI)); // Not sure whether std::sqrt is slower or std::pow. Todo: move sqrt out of the log?
};

void ExampleDistribution2::phi(korali::Sample &k)
{
  std::vector<double> hyperparams = k["Hyperparameters"];
  double sigma = hyperparams[_p.nClusters * _p.nDimensions];

  if (hyperparams.size() != _p.nDimensions * _p.nClusters + 1)
  {
    fprintf(stderr, "[Error] Hyperparameters should be one mean vector per cluster, plus a 1D variable sigma. The dimension of the hyperparameter vector did not match this.");
    exit(-1);
  }

  std::vector<std::vector<double>> mus(0);
  for (size_t i = 0; i < _p.nClusters; i++)
  {
    std::vector<double>::const_iterator first = hyperparams.begin() + i * _p.nDimensions;
    std::vector<double>::const_iterator last = hyperparams.begin() + (i + 1) * _p.nDimensions;
    std::vector<double> mu(first, last);
    if (mu.size() != _p.nDimensions)
    {
      fprintf(stderr, "[Error] Implementation error, dimensions did not match");
      exit(-1);
    }
    mus.push_back(mu);
  }
  /* For two variables:
        phi(sigma, mu1, mu2) = vec(1, -|mu1|^2, -|mu2|^2, mu1^T, mu2^T) * 1/(2*sigma)  */
  int phi_dim = 1 + _p.nDimensions * _p.nClusters + _p.nClusters;
  std::vector<double> phi(phi_dim);
  phi[0] = 1.0;
  std::vector<double> mu_x_two(mus[0].size());
  for (size_t i = 0; i < _p.nClusters; i++)
  {
    phi[i + 1] = -l2_norm_squared(mus[i]);
    // mu * 2:
    std::transform(mus[i].begin(), mus[i].end(), mu_x_two.begin(), [](auto &c) { return c * 2.0; });
    std::copy_n(mu_x_two.begin(), mu_x_two.size(), &phi[1 + _p.nClusters + i * _p.nDimensions]);
  }
  // *1/(2*sigma^2):
  std::transform(phi.begin(), phi.end(), phi.begin(), [&sigma](auto &c) { return c / (2 * sigma * sigma); });
  k["phi"] = phi;
};

#endif
