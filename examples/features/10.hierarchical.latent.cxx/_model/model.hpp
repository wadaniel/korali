#ifndef _MODEL_HPP_
#define _MODEL_HPP_

#include "korali.hpp"

#include "load_data.hpp"
#include "utils.hpp"

#include <cmath>
#include <random>
#include <vector>

class HierarchicalDistribution4
{
  public:
  HierarchicalDistribution4();
  void conditional_p(korali::Sample &k, std::vector<std::vector<double>> points = {}, bool internalData = false);
  pointsInfoStruct _p;
};

class HierarchicalDistribution5
{
  public:
  HierarchicalDistribution5();
  void conditional_p(korali::Sample &k, std::vector<std::vector<double>> points = {}, bool internalData = false);
  pointsInfoStructAdvanced _p;
};

#endif
