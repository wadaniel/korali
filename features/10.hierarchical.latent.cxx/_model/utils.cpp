#ifndef _UTILS_CPP_
#define _UTILS_CPP_

#include "utils.hpp"

double multivariate_gaussian_probability(const std::vector<std::vector<double>> &mus, int nDimensions,
                                         const std::vector<int>& assignments, int nClusters, double sigma,
                                         const std::vector<std::vector<double>>& points)
{
  /* Normalized, no log, clean pretty probability for N points. */
  assert(sigma > 0.0);
  assert(nDimensions > 0);
  for (int a : assignments)
    assert((0 <= a) && (a < nClusters));
  for (std::vector<double> v : points)
    assert(v.size() == nDimensions);

  int N = points.size();

  double p = 1.0;
  for (int i = 0; i < N; i++)
  {
    // get the cluster mean
    int c = assignments[i];
    std::vector<double> mu = mus[c];
    // update p
    p = p * univariate_gaussian_probability(mu, sigma, points[i]);
  }
  return p;
}

double univariate_gaussian_probability(const std::vector<double>& mu, double sigma, const std::vector<double>& point)
{
  assert(mu.size() == point.size());
  std::vector<double> distance(mu.size());
  std::transform(mu.begin(), mu.end(), point.begin(), distance.begin(), std::minus<double>());
  double squared_distance = std::inner_product(std::begin(distance), std::end(distance), std::begin(distance), 0.0);
  double exponent = -squared_distance / (2.0 * std::pow(sigma, 2.0));
  double denominator = (std::sqrt(2.0 * M_PI) * sigma);
  double probability = std::exp(exponent) / denominator;
  return probability;
}

#endif
