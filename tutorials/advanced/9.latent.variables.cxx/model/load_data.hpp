#ifndef _LOADDATA_H_
#define _LOADDATA_H_
#include "korali.hpp"

#include <stdlib.h>
#include <vector>

typedef struct pointsInfoStruct {
  std::vector<std::vector<double>> points;
  std::vector<int> assignments;
  size_t nPoints;
  size_t nDimensions;
  size_t nClusters;
 // std::vector<double> refTemp;
} pointsInfo;

pointsInfoStruct& univariateData();
pointsInfoStruct& multivariateData();

void reset_points(pointsInfoStruct& __p, std::vector<std::vector<double>> new_points, std::vector<int> new_assignments, int new_nClusters );

#endif // _HEAT2D_H_
