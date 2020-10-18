//  Korali model for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <random>

void runEnvironment(korali::Sample &s);
void createWorkEnvironment(size_t envId);

extern "C" void nek_init_(int* comm);
extern "C" void nek_solve_();
extern "C" void nek_end_();
