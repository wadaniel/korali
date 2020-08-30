#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

double multivariate_gaussian_probability(const std::vector<std::vector<double>> &mus,
                                         int nDimensions,
                                         const std::vector<int> &assignments,
                                         int nClusters,
                                         double sigma,
                                         const std::vector<std::vector<double>> &points);

double univariate_gaussian_probability(const std::vector<double> &mu,
                                       double sigma,
                                       const std::vector<double> &point);

#endif
