
#include "load_data.hpp"
#include "string.h"
#include <limits>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

pointsInfo __p;

pointsInfoStruct &univariateData()
{
  FILE *problemFile;
  FILE *problemFile2;
  unsigned long nPoints = 0;
  unsigned long nDimensions = 0;

  printf("Loading data from data_single.in ... \n");
  problemFile = fopen("_model/data_single.in", "r");
  if (!problemFile)
  {
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    std::cout << "Current workdir: " << cwd << std::endl;
    fprintf(stderr, "[Error] File not found: model/data_single.in. Make sure you run this script from the 'tutorials/advanced/9.latent.variables' directory. ");
    exit(-1);
  }
  problemFile2 = fopen("data_single.in", "r");
  fscanf(problemFile, "%lu", &nPoints);
  fscanf(problemFile, "%lu", &nDimensions);

  __p.points.resize(nPoints);
  __p.assignments.resize(0);

  for (size_t i = 0; i < nPoints; i++)
  {
    __p.points[i].resize(nDimensions);
    double val;
    for (size_t j = 0; j < nDimensions; j++)
    {
      fscanf(problemFile, "%le ", &val);
      __p.points[i][j] = val;
    }
  }
  fclose(problemFile);
  __p.nDimensions = nDimensions;
  __p.nPoints = nPoints;

  return __p;
};

pointsInfoStruct &multivariateData()
{
  // Also contains assignment information; remove to be sure you are not using it
  FILE *problemFile;

  unsigned long nPoints = 0;
  unsigned long nDimensions = 0;
  unsigned long nClusters = 0;

  printf("Loading data from data_multiple.in ... \n");
  problemFile = fopen("_model/data_multiple.in", "r");
  fscanf(problemFile, "%lu", &nPoints);
  fscanf(problemFile, "%lu", &nDimensions);
  fscanf(problemFile, "%lu", &nClusters);

  __p.points.resize(nPoints);
  __p.assignments.resize(nPoints);
  for (std::vector<double> &p : __p.points)
  {
    p.resize(nDimensions);
  }
  for (size_t i = 0; i < nPoints; i++)
  {
    __p.points[i].resize(nDimensions);
  }

  for (size_t i = 0; i < nPoints; i++)
  {
    double val_d;
    for (size_t j = 0; j < nDimensions; j++)
    {
      fscanf(problemFile, "%le ", &val_d);
      __p.points[i][j] = val_d;
    }
    unsigned long val;
    fscanf(problemFile, "%lu ", &val);
    __p.assignments[i] = val;
  }
  __p.nDimensions = nDimensions;
  __p.nPoints = nPoints;
  __p.nClusters = nClusters;

  fclose(problemFile); // Todo: remove some day. Added this when looking for a very mysterious bug

  return __p;
};

void reset_points(pointsInfoStruct &__p, std::vector<std::vector<double>> new_points, std::vector<int> new_assignments, int new_nClusters)
{
  assert(new_points.size() == new_assignments.size());
  __p.nPoints = new_points.size();
  __p.points = new_points;
  __p.assignments = new_assignments;
  for (std::vector<double> pt : new_points)
    assert(pt.size() == __p.nDimensions && "Cannot change dimensionality (could implement it though)");
  __p.nClusters = new_nClusters;
  return;
}
