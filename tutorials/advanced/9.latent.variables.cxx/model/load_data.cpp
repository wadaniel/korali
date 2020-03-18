
#include <stdio.h>
#include <math.h>
#include <limits>
#include "load_data.hpp"
#include "string.h"
#include <unistd.h>

pointsInfo __p;

pointsInfoStruct& univariateData()
{

 FILE *problemFile;
 FILE *problemFile2;
 uint32_t nPoints = 0;
 int nDimensions = 0;

 printf("Loading data from data_single.in ... \n");// @suppress("Function cannot be resolved")
 problemFile = fopen("model/data_single.in", "r");// @suppress("Function cannot be resolved")
 if (!problemFile){
	 	 char cwd[PATH_MAX];
	 	 getcwd(cwd, sizeof(cwd));
	 	 std::cout << "Current workdir: " << cwd << std::endl;
		 korali::logError("File not found: model/data_single.in. Make sure you run this script from the 'tutorials/advanced/9....' directory. ");}
 problemFile2 = fopen("data_single.in", "r");// @suppress("Function cannot be resolved")
 fscanf(problemFile, "%lu", &nPoints); // @suppress("Function cannot be resolved")
 fscanf(problemFile, "%lu", &nDimensions); // @suppress("Function cannot be resolved")

 __p.points.resize(nPoints);
 __p.assignments.resize(0);

 for (size_t i = 0; i < nPoints; i++)
 {
  __p.points[i].resize(nDimensions);
  double val;
  for (size_t j = 0; j < nDimensions; j++){
      fscanf(problemFile, "%le ", &val);
       __p.points[i][j] = val;
  }
 }
 fclose(problemFile);
 __p.nDimensions = nDimensions;
 __p.nPoints = nPoints;

 return __p;
}



pointsInfoStruct& multivariateData()
{
 // Also contains assignment information; remove to be sure you are not using it
 FILE *problemFile;

 size_t nPoints = 0;
 size_t nDimensions = 0;
 size_t nClusters = 0;

 printf("Loading data from data_multiple.in ... \n");
 problemFile = fopen("model/data_single.in", "r");
 fscanf(problemFile, "%lu", &nPoints); // @suppress("Function cannot be resolved")
 fscanf(problemFile, "%lu", &nDimensions);
 fscanf(problemFile, "%lu", &nClusters);

 __p.points.resize(nPoints );
 __p.assignments.resize(nPoints);
 for (std::vector<double> &p : __p.points){
     p.resize(nDimensions);
 }

 for (size_t i = 0; i < nPoints; i++)
 {
  double val;
  for (size_t j = 0; j < nDimensions; j++){
      fscanf(problemFile, "%le ", &val);
      __p.points[i][j] = val;
  }
  fscanf(problemFile, "%le ", &val); __p.assignments[i] = val;
 }
 __p.nDimensions = nDimensions;
 __p.nPoints = nPoints;

 return __p;
}

