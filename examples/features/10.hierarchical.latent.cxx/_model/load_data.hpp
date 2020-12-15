#ifndef _LOADDATA_H_
#define _LOADDATA_H_
#include "korali.hpp"

#include <stdlib.h>
#include <vector>

typedef struct pointsInfoStruct
{
  std::vector<std::vector<double>> data;
  size_t nIndividuals;
  size_t nSamplesEach;
  double sigma;
  double omega;
} pointsInfo;

typedef struct pointsInfoStructAdvanced
{
  std::vector<std::vector<std::vector<double>>> data;
  size_t nIndividuals;
  size_t nDimensions;
  size_t maxNSamples;
  std::vector<size_t> nSamplesEach;
  double sigma;
  double dNormal;
  double dLognormal;
  double dLogitnormal;
} pointsInfoAdv;

pointsInfoStruct &simplePopulationData();
pointsInfoStructAdvanced &populationData();

void reset_points(pointsInfoStruct &__p, std::vector<std::vector<double>> new_data, int new_nSamplesEach, double new_sigma, double new_omega);
void reset_points_adv(pointsInfoStructAdvanced &__p, std::vector<std::vector<double>> new_data, double new_sigma);

#endif // _HEAT2D_H_
