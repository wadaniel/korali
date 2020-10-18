//  Korali Environment for Nek5000
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#ifndef _NEK5000_ENVIRONMENT_HPP
#define _NEK5000_ENVIRONMENT_HPP

#define STATE_SIZE 6
#define ACTION_SIZE 1

#include "korali.hpp"
#include <random>

extern void runEnvironment(korali::Sample &s);
extern void updateAgent();

extern korali::Sample* sample;
extern double _reward;
extern double _state[STATE_SIZE];
extern double _action[ACTION_SIZE];

extern "C"
{
 void resetenv_();
 void nek_init_(int* comm);
 void nek_solve_();
 void nek_end_();
 void setreward_(double* reward);
 void setstate_(double state[STATE_SIZE]);
 void getaction_(double action[ACTION_SIZE]);
 void updateagent_();
}

void setreward_(double* reward) { _reward = *reward; }
void setstate_(double state[STATE_SIZE]) { for (int i = 0; i < STATE_SIZE; i++) _state[i] = state[i]; }
void getaction_(double action[ACTION_SIZE]) { for (int i = 0; i < ACTION_SIZE; i++) _action[i] = action[i]; }
void updateagent_() { updateAgent(); }

#endif
