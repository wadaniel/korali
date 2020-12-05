//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"

#ifndef TEST

Simulation *_environment;
bool _isTraining;
std::mt19937 _randomGenerator;
size_t _maxSteps;
Shape *_object;
StefanFish *_agent;

void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t seed = s["Sample Id"];
  _randomGenerator.seed(seed);

  // Reseting environment and setting initial conditions
  _environment->reset();
  setInitialConditions(_agent, _object);

  // Setting initial state
  auto state = _agent->state(_object);
  s["State"] = state;

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Starting main environment loop
  bool done = false;
  while (done == false && curStep < _maxSteps)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action
    _agent->act(t, action);

    // Run the simulation until next action is required
    tNextAct += _agent->getLearnTPeriod() * 0.5;
    while (t < tNextAct)
    {
      const double dt = _environment->calcMaxTimestep();
      t += dt;

      // Advance simulation and check whether it is correct
      if (_environment->advance(dt))
      {
        fprintf(stderr, "Error during environment\n");
        exit(-1);
      }

      // Check if simulation is done.
      done = isTerminal(_agent, _object);
    }

    // Reward is -10 if state is terminal; otherwise obtain it from the agent's efficiency
    double reward = done ? -10.0 : _agent->EffPDefBnd;

    // Printing Information:
    //    printf("[Korali] -------------------------------------------------------\n");
    //    printf("[Korali] Step: %lu/%lu\n", curStep, _maxSteps);
    //    printf("[Korali] State: [ %.3f", state[0]);
    //    for (size_t i = 1; i < state.size(); i++) printf(", %.3f", state[i]);
    //    printf("]\n");
    printf("[Korali] Step: %lu/%lu, Action: [ %.3f", curStep, _maxSteps, action[0]);
    for (size_t i = 1; i < action.size(); i++) printf(", %.3f", action[i]);
    printf("]\n");
    ////    printf("[Korali] Terminal: %d\n", done);
    ////    printf("[Korali] -------------------------------------------------------\n");

    // Obtaining new agent state
    state = _agent->state(_object);

    // Storing reward
    s["Reward"] = reward;

    // Storing new state
    s["State"] = state;

    // Advancing to next step
    curStep++;
  }

  // Setting finalization status
  if (done == true)
    s["Termination"] = "Normal";
  else
    s["Termination"] = "Truncated";
}

void initializeEnvironment(int argc, char *argv[])
{
  _maxSteps = 200;

  _environment = new Simulation(argc, argv);
  _environment->init();

  _object = _environment->getShapes()[0];
  _agent = dynamic_cast<StefanFish *>(_environment->getShapes()[1]);
  if (_agent == nullptr)
  {
    fprintf(stderr, "[Error] Agent was not a StefanFish!\n");
    exit(-1);
  }
}

void setInitialConditions(StefanFish *a, Shape *p)
{
  std::uniform_real_distribution<double> disA(-20. / 180. * M_PI, 20. / 180. * M_PI);
  std::uniform_real_distribution<double> disX(0, 0.5), disY(-0.25, 0.25);

  const double SX = _isTraining ? disX(_randomGenerator) : 0.35;
  const double SY = _isTraining ? disY(_randomGenerator) : 0.00;
  const double SA = _isTraining ? disA(_randomGenerator) : 0.00;

  double C[2] = {p->center[0] + (1 + SX) * a->length, p->center[1] + SY * a->length};
  p->centerOfMass[1] = p->center[1] - (C[1] - p->center[1]);
  p->center[1] = p->center[1] - (C[1] - p->center[1]);
  a->setCenterOfMass(C);
  a->setOrientation(SA);
}

bool isTerminal(StefanFish *a, Shape *p)
{
  const double X = (a->center[0] - p->center[0]) / a->length;
  const double Y = (a->center[1] - p->center[1]) / a->length;
  assert(X > 0);
  // cylFollow
  return std::fabs(Y) > 1 || X < 0.5 || X > 3;
  // extended follow
  // return std::fabs(Y)>1 || X<1 || X>3;
  // restricted follow
  //return std::fabs(Y) > 0.75 || X < 1 || X > 2;
}

#else

// Environment for configuration test only
void runEnvironment(korali::Sample &s)
{
  fprintf(stderr, "[Warning] Using test-only setup. If you want to run the actual experiment, run ./install_deps.sh first and re-compile.\n");

  for (size_t i = 0; i < 10; i++)
  {
    s["State"] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    s.update();
    s["Reward"] = -10.0;
  }
  s["Termination"] = "Normal";
}

#endif
