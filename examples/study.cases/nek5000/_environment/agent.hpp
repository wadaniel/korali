//  Korali Agent for Nek5000
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#ifndef _NEK5000_AGENT_HPP
#define _NEK5000_AGENT_HPP

#ifndef TEST

#define STATE_SIZE 6
#define ACTION_SIZE 1

#include "korali.hpp"

extern void updateAgent();

extern korali::Sample *sample;
extern double _reward;
extern double _state[STATE_SIZE];
extern double _action[ACTION_SIZE];

extern "C"
{
  void setreward_(double *reward);
  void setstate_(double state[STATE_SIZE]);
  void getaction_(double action[ACTION_SIZE]);
  void updateagent_();
}

void setreward_(double *reward) { _reward = *reward; }
void setstate_(double state[STATE_SIZE])
{
  for (int i = 0; i < STATE_SIZE; i++) _state[i] = state[i];
}
void getaction_(double action[ACTION_SIZE])
{
  for (int i = 0; i < ACTION_SIZE; i++) _action[i] = action[i];
}
void updateagent_() { updateAgent(); }

#endif // TEST

#endif // _NEK5000_AGENT_HPP
