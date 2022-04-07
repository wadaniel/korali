//  Korali environment for CubismUP-2D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"
#include <Cubism/ArgumentParser.h>

#define NACTIONS 3

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);

bool isTerminal(StefanFish *agent);
double getReward(StefanFish *agent);
std::vector<double> getState(StefanFish *agent);
Simulation * initializeEnvironment(korali::Sample &s, const int task);
