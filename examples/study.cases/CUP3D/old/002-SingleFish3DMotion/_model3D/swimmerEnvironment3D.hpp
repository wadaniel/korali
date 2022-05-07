//  Korali environment for CubismUP-3D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "obstacles/StefanFish.h"
#include "Simulation.h"
#include "utils/BufferedLogger.h"
#include <Cubism/ArgumentParser.h>

#define NACTIONS 2
using namespace cubismup3d;
// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
void setInitialConditions(cubismup3d::StefanFish *agent, size_t agentId, const bool isTraining, int rank, MPI_Comm comm);
bool isTerminal(cubismup3d::StefanFish *agent);
double getReward(cubismup3d::StefanFish *agent);
std::vector<double> getState(cubismup3d::StefanFish *agent);
Simulation * initializeEnvironment(korali::Sample &s, const int task);
