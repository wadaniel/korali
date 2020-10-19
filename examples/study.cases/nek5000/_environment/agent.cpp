//  Korali Environment for Nek5000
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "agent.hpp"
#include "unistd.h"
#include "stdio.h"

korali::Sample* sample;
double _reward;
double _state[STATE_SIZE];
double _action[ACTION_SIZE];

void updateAgent()
{
 // Setting states
 for (size_t i = 0; i < STATE_SIZE; i++)
  (*sample)["State"][i] = _state[i];

 // Setting Reward
 (*sample)["Reward"] = _reward;

 // Getting new action
 sample->update();

 // Reading new action
 for (size_t i = 0; i < ACTION_SIZE; i++)
  _action[i] = (*sample)["Action"][i];

 // Printing step info
 printf("[Korali] ----------------------------------------------------\n");
 printf("[Korali] Step Information:\n");
 printf("[Korali] State: [ %f, %f, %f, %f, %f, %f ]\n", _state[0], _state[1], _state[2], _state[3], _state[4], _state[5]);
 printf("[Korali] Reward: %f\n", _reward);
 printf("[Korali] Action: %f\n", _action[0]);
 printf("[Korali] ----------------------------------------------------\n");
}

