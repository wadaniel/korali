#ifndef _LOADDATA_H_
#define _LOADDATA_H_
#include "korali.hpp"

#include <stdlib.h>
#include <vector>

typedef struct pointsInfoStruct {
  std::vector<double> points;
  std::vector<double> assignments;
  size_t nPoints;
  size_t nDimensions;
 // std::vector<double> refTemp;
} pointsInfo;

pointsInfoStruct& univariateData();
pointsInfoStruct& multivariateData();

#endif // _HEAT2D_H_
