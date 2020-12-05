//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>

void runEnvironment(korali::Sample &s);

#ifndef TEST

  #include "Obstacles/StefanFish.h"
  #include "Simulation.h"

void initializeEnvironment();
void setInitialConditions(StefanFish *a, Shape *p);
bool isTerminal(StefanFish *a, Shape *p);

// Global variables for the simulation (ideal if this would be a class instead)
extern bool _initialized;
extern Simulation *_environment;
extern bool _isTraining;
extern std::mt19937 _randomGenerator;
extern size_t _maxSteps;
extern Shape *_object;
extern StefanFish *_agent;

#endif
