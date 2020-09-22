//  Korali model for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "model.hpp"

Simulation* _simulation;
bool __isTraining;
std::mt19937 _randomGenerator;
size_t _maxSteps;
Shape* _object;
StefanFish* _agent;

void runEnvironment(korali::Sample &s)
{
 size_t seed = s["Sample Id"];
 _randomGenerator.seed(seed);

 s["State"] = { 0.0 }; // Get initial state
 size_t curStep = 0;
 bool done = false; // flag to check whether simulation is done

 while(done == false && curStep < _maxSteps)
 {
  // Getting new action
  s.update();

  // Reading new action
  std::vector<double> action = s["Action"][0];

  // Performing action
  //state, reward, done, info = cart.step(action)
  double reward = 0.0;
  std::vector<double> state = { 1.0 };

  // Storing reward
  s["Reward"] = reward;

  // Storing new state
  s["State"] = { state };

  // Check termination
  done = true;
  curStep++;
 }
}

void initializeEnvironment(int argc, char* argv[])
{
 printf("Configuring Cubism2D with the following parameters:\n");
 for(int i = 0; i < argc; i++) printf("arg: %s\n", argv[i]);

 const unsigned _maxSteps = 200;

 _simulation = new Simulation(argc, argv);
 _simulation->init();

 _object = _simulation->getShapes()[0];
 _agent = dynamic_cast<StefanFish*>(_simulation->getShapes()[1] );
 if(_agent == nullptr) { printf("Agent was not a StefanFish!\n"); exit(-1); }
}

inline void resetIC(StefanFish* const a, Shape*const p)
{
  std::uniform_real_distribution<double> disA(-20./180.*M_PI, 20./180.*M_PI);
  std::uniform_real_distribution<double> disX(0, 0.5),  disY(-0.25, 0.25);

  const double SX = _isTraining ? disX(_randomGenerator) : 0.35;
  const double SY = _isTraining ? disY(_randomGenerator) : 0.00;
  const double SA = _isTraining ? disA(_randomGenerator) : 0.00;

  double C[2] = { p->center[0] + (1+SX)*a->length, p->center[1] + SY*a->length };
  p->centerOfMass[1] = p->center[1] - ( C[1] - p->center[1] );
  p->center[1] = p->center[1] - ( C[1] - p->center[1] );
  a->setCenterOfMass(C);
  a->setOrientation(SA);
}
