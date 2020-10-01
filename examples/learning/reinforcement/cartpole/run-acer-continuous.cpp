#include "_model/environment.hpp"

int main(int argc, char *argv[])
{
  auto k = korali::Engine();
  auto e = korali::Experiment();

  /// Defining the Cartpole problem's configuration

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
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
  e["Variables"][5]["Lower Bound"] = -10.0;
  e["Variables"][5]["Upper Bound"] = +10.0;
  e["Variables"][5]["Noise Sigma"] = 0.5;

  /// Configuring ACER hyperparameters

  e["Solver"]["Type"] = "Agent / ACER / Continuous";
  e["Solver"]["Importance Weight Truncation"] = 5.0;
  e["Solver"]["Trajectory Size"] = 1000;
  e["Solver"]["Discount Factor"] = 0.995;
  e["Solver"]["Off Policy Updates"] = 8;
  e["Solver"]["Optimization Steps Per Trajectory"] = 1;

  /// Defining Experience Replay configuration

  e["Solver"]["Experience Replay"]["Start Size"] = 1500;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 500000;

  // Defining Q-Critic and Action-selection (policy) optimizers

  e["Solver"]["Critic"]["Optimizer"]["Type"] = "Optimizer/Adam";
  e["Solver"]["Critic"]["Optimizer"]["Eta"] = 0.001;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Node Count"] = 5;
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

  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000000001;

  e["Solver"]["Normalization Steps"] = 32;
  e["Solver"]["Normalization Batch Size"] = 32;

  // Defining Policy Configuration

  e["Solver"]["Policy"]["Trust Region"]["Enabled"] = true;
  e["Solver"]["Policy"]["Trust Region"]["Divergence Constraint"] = 1.0;
  e["Solver"]["Policy"]["Trust Region"]["Adoption Rate"] = 0.99;

  e["Solver"]["Policy"]["Optimizer"]["Type"] = "Optimizer/Adam";
  e["Solver"]["Policy"]["Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = -1.0;
  e["Solver"]["Policy"]["Optimizer"]["Eta"] = 0.001;
  e["Solver"]["Policy"]["Sample Population"] = 10;

  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Node Count"] = 5;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Alpha"] = 0.0;

  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Alpha"] = 0.0;

  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Node Count"] = 1;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Logistic";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000000001;

  e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = { 10.0 };

  /// Defining Termination Criteria

  e["Solver"]["Training Reward Threshold"] = 490;
  e["Solver"]["Policy Testing Episodes"] = 20;
  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490;

  /// Setting file output configuration

  e["File Output"]["Frequency"] = 100000;

  /// Running Experiment

  k.run(e);

  return 0;
}
