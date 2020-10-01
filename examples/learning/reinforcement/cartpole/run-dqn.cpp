#include "_model/cartpole.hpp"
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
  done = cart.advance(s["Action"]);

  // Getting Reward
  s["Reward"] = cart.getReward();

  // Storing New State
  s["State"] = cart.getState();

  // Advancing step counter
  step = step + 1;
 }
}

int main(int argc, char *argv[])
{
  auto k = korali::Engine();
  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Reinforcement Learning / Discrete";
  e["Problem"]["Possible Actions"] = { { -10.0 }, { -9.0 }, { -8.0 }, { -7.0 }, { -6.0 }, { -5.0 }, { -4.0 }, { -3.0 }, { -2.0 }, { -1.0 }, {0.0},
                                       {  10.0 }, {  9.0 }, {  8.0 }, {  7.0 }, {  6.0 }, {  5.0 }, {  4.0 }, {  3.0 }, {  2.0 }, {  1.0 }  };
  e["Problem"]["Environment Function"] = &env;
  e["Problem"]["Action Repeat"] = 1;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  e["Variables"][0]["Name"] = "Cart Position";
  e["Variables"][0]["Type"] = "State";

  e["Variables"][1]["Name"] = "Cart Velocity";
  e["Variables"][1]["Type"] = "State";

  e["Variables"][2]["Name"] = "Pole Omega";
  e["Variables"][2]["Type"] = "State";

  e["Variables"][3]["Name"] = "Pole Cos(Angle)";
  e["Variables"][3]["Type"] = "State";

  e["Variables"][4]["Name"] = "Pole Sin(Angle)";
  e["Variables"][4]["Type"] = "State";

  e["Variables"][5]["Name"] = "Force";
  e["Variables"][5]["Type"] = "Action";

  /// Configuring DQN hyperparameters

  e["Solver"]["Type"] = "Agent/DQN";
  e["Solver"]["Trajectory Size"] = 1;
  e["Solver"]["Optimization Steps Per Trajectory"] = 1;

  /// Defining Experience Replay configuration

  e["Solver"]["Experience Replay"]["Start Size"] = 500;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 150000;

  /// Defining training policy configuration

  e["Solver"]["Policy"]["Epsilon"]["Initial Value"] = 1.0;
  e["Solver"]["Policy"]["Epsilon"]["Target Value"] = 0.05;
  e["Solver"]["Policy"]["Epsilon"]["Decrease Rate"] = 0.05;

  // Defining Q-Critic and Action-selection (policy) optimizers

  e["Solver"]["Critic"]["Mini Batch Size"] = 32;
  e["Solver"]["Critic"]["Discount Factor"] = 0.99;
  e["Solver"]["Critic"]["Optimizer"]["Type"] = "Optimizer/Adam";
  e["Solver"]["Critic"]["Optimizer"]["Eta"] = 0.01;

  /// Defining the shape of the neural network

  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Node Count"] = 6;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = false;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Node Count"] = 1;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Normalization Steps"] = 32;

  /// Defining Termination Criteria

  e["Solver"]["Training Reward Threshold"] = 490;
  e["Solver"]["Policy Testing Episodes"] = 20;
  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490;

  /// Setting file output configuration

  e["File Output"]["Frequency"] = 100;

  /// Running Experiment

  k.run(e);

  return 0;
}
