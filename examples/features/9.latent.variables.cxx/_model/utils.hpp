#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

double multivariate_gaussian_probability(std::vector<std::vector<double>> mus, int nDimensions, std::vector<int> assignments, int nClusters, double sigma, std::vector<std::vector<double>> points);
double univariate_gaussian_probability(std::vector<double> mu, double sigma, std::vector<double> point);

double l2_norm(std::vector<double> const &u);
double l2_norm_squared(std::vector<double> const &u);

std::vector<double> normalize(std::vector<double> const &u);

template <typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> data)
{
  /* From: https://stackoverflow.com/questions/6009782/how-to-pivot-a-vector-of-vectors */
  // this assumes that all inner vectors have the same size and
  // allocates space for the complete result in advance
  std::vector<std::vector<T>> result(data[0].size(),
                                     std::vector<T>(data.size()));
  for (size_t i = 0; i < data[0].size(); i++)
  {
    for (size_t j = 0; j < data.size(); j++)
    {
      result[i][j] = data[j][i];
    }
  }
  return result;
}

#endif
