#include "cartpole.hpp"
#include "korali.hpp"

#define MAX_STEPS 500
CartPole cart;

void env(korali::Sample& s)
{
 // Setting seed
 size_t seed = s["Sample Id"];
 _randomGenerator.seed(seed);

 // Initializing environment
 cart.reset();
 s["State"] = cart.getState();
 size_t step = 0;
 bool done = false;

 while (done == false and step < MAX_STEPS)
 {
  // Getting new action
  s.update();

  // Performing the action
  printf("Action: %.2f\n", s["Action"].get<std::vector<double>>()[0]);
  done = cart.advance(s["Action"]);

  // Getting Reward
  s["Reward"] = cart.getReward();

  // Storing New State
  s["State"] = cart.getState();

  // Advancing step counter
  step = step + 1;
 }
}
