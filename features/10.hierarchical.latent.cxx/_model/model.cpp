#ifndef _MODEL_CPP_
#define _MODEL_CPP_

#include "korali.hpp"

#include "load_data.hpp"
#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <typeinfo>
#include <vector>

/*
Model 4:
 (see python tutorial, in basic/...)
*/

HierarchicalDistribution4::HierarchicalDistribution4()
{
  _p = simplePopulationData();
};

//     See also the python tutorial, in basic/... . Here we only know the part
//         p(data | latent), which is a product of Gaussians:
//
//          Model 3:
//            draw psi_i ~ N(theta, omega**2)
//            draw x_i ~ N(psi_i, sigma**2)
//
void HierarchicalDistribution4::conditional_p(korali::Sample &s, std::vector<std::vector<double>> points, bool internalData )
{
  std::vector<double> latentVariables = s["Latent Variables"];
  assert(latentVariables.size() == 1); // nr latent space dimensions = 1
  if (internalData){
    assert (points.size() == 0);
    points = s["Data Points"].get<std::vector<std::vector<double>>>();
  }
  else assert (points.size() > 0);

  double sigma = _p.sigma;

  // log(p(data | mean, sigma ))
  double logp = 0;
//  for (size_t i=0; i < _p.nIndividuals; i++){
    std::vector<double> pt = {points[0][0]}; // in this example there is only one point per individual
    double p = univariate_gaussian_probability(latentVariables, sigma, pt);

    logp += log(p);
//   }

  s["logLikelihood"] = logp;
};

/*
Model 5:
 - multiple dimensions
 - multiple distribution types
 - latent variable coordinates are correlated
 - p(datapoint | latent) is still a normal distribution N(latent, sigma**2)
*/
HierarchicalDistribution5::HierarchicalDistribution5()
{
  _p = populationData();
};

void HierarchicalDistribution5::conditional_p(korali::Sample &s, std::vector<std::vector<double>> points, bool internalData )
{
  std::vector<double> latentVariables = s["Latent Variables"];
  assert(latentVariables.size() == _p.nDimensions);
  if (internalData){
    assert(points.size() == 0);
    points = s["Data Points"].get<std::vector<std::vector<double>>>();
  }
  else assert(points.size() > 0);


  double sigma = _p.sigma;

  // log(p(data | mean=latent variable, sigma ))
  double logp = 0;
  for (size_t j = 0; j < points.size() ; j++){
    assert(points[j].size() == _p.nDimensions);
    for (size_t i = 0; i < _p.nDimensions; i++)
    {
      double pt = points[j][i];
      double mean = latentVariables[i];
      std::vector<double> pt_vec({pt});
      std::vector<double> mean_vec({mean});
      double p = univariate_gaussian_probability(mean_vec, sigma, pt_vec);
      logp += log(p);
    }
  }

  s["logLikelihood"] = logp;
};

#endif
