//  Korali model for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <random>

void runEnvironment(korali::Sample &s);
void initializeEnvironment();

extern "C" void nek_init_(int* comm);
extern "C" void nek_solve_(int* comm);
extern "C" void nek_end_(int* comm);

extern std::mt19937 _randomGenerator;

