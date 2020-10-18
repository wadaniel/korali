#include "_model/model.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  initializeEnvironment();

  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Action Repeat"] = 1;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  e["Variables"][0]["Name"] = "Some State Variable 1";
  e["Variables"][0]["Type"] = "State";

  e["Variables"][1]["Name"] = "Some State Variable 2";
  e["Variables"][1]["Type"] = "State";

  e["Variables"][2]["Name"] = "Some State Variable 3";
  e["Variables"][2]["Type"] = "State";

  e["Variables"][3]["Name"] = "Some State Variable 4";
  e["Variables"][3]["Type"] = "State";

  e["Variables"][4]["Name"] = "Some State Variable 5";
  e["Variables"][4]["Type"] = "State";

  e["Variables"][5]["Name"] = "Some Action";
  e["Variables"][5]["Type"] = "Action";
  e["Variables"][5]["Lower Bound"] = 0.0;
  e["Variables"][5]["Upper Bound"] = 0.01;
  e["Variables"][5]["Exploration Noise"]["Enabled"] = true;
  e["Variables"][5]["Exploration Noise"]["Distribution"]["Type"] = "Univariate/Normal";
  e["Variables"][5]["Exploration Noise"]["Distribution"]["Mean"] = 0.0;
  e["Variables"][5]["Exploration Noise"]["Distribution"]["Standard Deviation"] = 0.0005;
  e["Variables"][5]["Exploration Noise"]["Theta"] = 0.05;

  ////// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent/SDPG";
  e["Solver"]["Normalization Steps"] = 32;
  e["Solver"]["Trajectory Size"] = 1;
  e["Solver"]["Optimization Steps Per Trajectory"] = 1;

  e["Solver"]["Random Action Probability"]["Initial Value"] = 0.01;
  e["Solver"]["Random Action Probability"]["Target Value"] = 0.01;
  e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.00;

  ////// Defining the configuration of replay memory

  e["Solver"]["Experience Replay"]["Start Size"] = 10000;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 262144;

  //// Defining Critic Configuration

  e["Solver"]["Critic"]["Learning Rate"] = 0.00001;
  e["Solver"]["Critic"]["Discount Factor"] = 0.99;
  e["Solver"]["Critic"]["Mini Batch Size"] = 256;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 128;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 128;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = true;

  //// Defining Policy Configuration

  e["Solver"]["Policy"]["Learning Rate"] = 0.000001;
  e["Solver"]["Policy"]["Mini Batch Size"] = 16;
  e["Solver"]["Policy"]["Adoption Rate"] = 0.80;

  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 128;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 128;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Logistic";

  e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = { 0.01 };

  ////// Defining Termination Criteria

  e["Solver"]["Training Reward Threshold"] = 1.0;
  e["Solver"]["Policy Testing Episodes"] = 20;
  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 1.3;

  ////// Setting file output configuration

  e["File Output"]["Frequency"] = 10000;

  ////// Running Experiment

  auto k = korali::Engine();
  k.run(e);
}
