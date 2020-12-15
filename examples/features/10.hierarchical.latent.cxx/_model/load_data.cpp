
#include "load_data.hpp"
#include "string.h"
#include <limits>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

pointsInfo __p;
pointsInfoAdv __pa;

pointsInfoStruct &simplePopulationData()
{
  FILE *problemFile;
  FILE *problemFile2;
  unsigned long nIndividuals = 0;
  unsigned long nSamplesEach = 0;
  double sigma = 0;
  double omega = 0;
  // std::string filenm = "data_925_test.in";
  std::string filenm = "data_925.in";

  printf("Loading data from %s ... \n", filenm.c_str());
  problemFile = fopen((std::string("_model/") + filenm).c_str(), "r");
  if (!problemFile)
  {
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    std::cout << "Current workdir: " << cwd << std::endl;
    fprintf(stderr, "[Error] File not found: model/data_925.in. Make sure you run this script from the 'tutorials/advanced/10.hierarchical.latent.cxx' directory. ");
    exit(-1);
  }
  problemFile2 = fopen(filenm.c_str(), "r");
  fscanf(problemFile, "%lu", &nIndividuals);
  fscanf(problemFile, "%lu", &nSamplesEach);
  fscanf(problemFile, "%le", &sigma);
  fscanf(problemFile, "%le", &omega);

  __p.data.resize(nIndividuals);

  for (size_t i = 0; i < nIndividuals; i++)
  {
    __p.data[i].resize(nSamplesEach);
    double val;
    for (size_t j = 0; j < nSamplesEach; j++)
    {
      fscanf(problemFile, "%le ", &val);
      __p.data[i][j] = val;
    }
  }
  fclose(problemFile);
  __p.nIndividuals = nIndividuals;
  __p.nSamplesEach = nSamplesEach;
  __p.sigma = sigma;
  __p.omega = omega;

  return __p;
};

void reset_points(pointsInfoStruct &__p, std::vector<std::vector<double>> new_data, int new_nSamplesEach, double new_sigma, double new_omega)
{
  __p.nIndividuals = new_data.size();
  __p.nSamplesEach = new_nSamplesEach;
  __p.sigma = new_sigma;
  __p.omega = new_omega;
  __p.data = new_data;
  for (std::vector<double> pt : new_data)
    assert(pt.size() == new_nSamplesEach && "... I think the inner dimension was over the samples per each individual. This is a primitive example with dimension == 1.");
  return;
}

pointsInfoStructAdvanced &populationData()
{
  FILE *problemFile;
  FILE *problemFile2;

  unsigned long nIndividuals = 0;
  unsigned long nDimensions = 0;
  unsigned long maxNSamples = 0;
  std::vector<unsigned long> nSamplesEach;
  double sigma = 0;
  unsigned long dNormal = 0;
  unsigned long dLognormal = 0;
  unsigned long dLogitnormal = 0;
  std::string filenm = "data_advanced.in";

  printf("Loading data from %s ... \n", filenm.c_str());
  problemFile = fopen((std::string("_model/") + filenm).c_str(), "r");
  if (!problemFile)
  {
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    std::cout << "Current workdir: " << cwd << std::endl;
    fprintf(stderr, "[Error] File not found: model/%s. Make sure you run this script from the 'features/10.hierarchical.latent.cxx' directory. ", filenm.c_str());
    exit(-1);
  }
  problemFile2 = fopen(filenm.c_str(), "r");
  fscanf(problemFile, "%lu", &nIndividuals);
  fscanf(problemFile, "%lu", &nDimensions);
  fscanf(problemFile, "%lu", &maxNSamples);
  fscanf(problemFile, "%le", &sigma);
  fscanf(problemFile, "%lu", &dNormal);
  fscanf(problemFile, "%lu", &dLognormal);
  fscanf(problemFile, "%lu", &dLogitnormal);

  __pa.data.resize(nIndividuals);
  nSamplesEach.resize(nIndividuals);
  unsigned long nSamples;
  char discardChar[2];
  double val;

  for (size_t i = 0; i < nIndividuals; i++)
  {
    fscanf(problemFile, "%1s ", discardChar);
    assert(discardChar[0] == 'N');
    fscanf(problemFile, "%lu ", &nSamples);
    __pa.data[i].resize(nSamples);
    nSamplesEach[i] = nSamples;
    for (size_t j = 0; j < nSamples; j++)
    {
      __pa.data[i][j].resize(nDimensions);
      for (size_t dim = 0; dim < nDimensions; dim++)
      {
        fscanf(problemFile, "%le ", &val);
        __pa.data[i][j][dim] = val;
      }
    }
  }
  fclose(problemFile);

  __pa.nIndividuals = nIndividuals;
  __pa.nDimensions = nDimensions;
  __pa.maxNSamples = maxNSamples;
  __pa.nSamplesEach = nSamplesEach;
  __pa.sigma = sigma;
  __pa.dNormal = dNormal;       //0
  __pa.dLognormal = dLognormal; //2
  __pa.dLogitnormal = dLogitnormal;

  return __pa;
};

void reset_points_adv(pointsInfoStructAdvanced &__p, std::vector<std::vector<double>> new_data, double new_sigma)
{
  //  __p.nIndividuals = new_data.size();
  //  __p.nSamplesEach = new_nSamplesEach;
  //  __p.sigma = new_sigma;
  //  __p.omega = new_omega;
  //  __p.data = new_data;
  //  for (std::vector<double> pt : new_data)
  //    assert(pt.size() == new_nSamplesEach && "... I think the inner dimension was over the samples per each individual. This is a primitive example with dimension == 1.");
  return;
}
