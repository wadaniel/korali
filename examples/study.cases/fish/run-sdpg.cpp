#include "_model/model.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  // Gathering actual arguments from MPI
  //MPI_Init(&argc, &argv);

  // Initializing environment
  initializeEnvironment(argc, argv);

  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Action Repeat"] = 1;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  // Setting up the 10 state variables
  size_t curVariable = 0;
  for(; curVariable < 10; curVariable++)
  {
   e["Variables"][curVariable]["Name"] = std::string("StateVar") + std::to_string(curVariable);
   e["Variables"][curVariable]["Type"] = "State";
  }

  e["Variables"][curVariable]["Name"] = "Curvature";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -1.0;
  e["Variables"][curVariable]["Upper Bound"] = +1.0;
  e["Variables"][curVariable]["Exploration Noise"]["Enabled"] = true;
  e["Variables"][curVariable]["Exploration Noise"]["Distribution"]["Type"] = "Univariate/Normal";
  e["Variables"][curVariable]["Exploration Noise"]["Distribution"]["Mean"] = 0.0;
  e["Variables"][curVariable]["Exploration Noise"]["Distribution"]["Standard Deviation"] = 0.01;
  e["Variables"][curVariable]["Exploration Noise"]["Theta"] = 0.05;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Force?";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -0.25;
  e["Variables"][curVariable]["Upper Bound"] = +0.25;
  e["Variables"][curVariable]["Exploration Noise"]["Enabled"] = true;
  e["Variables"][curVariable]["Exploration Noise"]["Distribution"]["Type"] = "Univariate/Normal";
  e["Variables"][curVariable]["Exploration Noise"]["Distribution"]["Mean"] = 0.0;
  e["Variables"][curVariable]["Exploration Noise"]["Distribution"]["Standard Deviation"] = 0.001;
  e["Variables"][curVariable]["Exploration Noise"]["Theta"] = 0.0125;

  ////// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent/SDPG";
  e["Solver"]["Normalization Steps"] = 32;
  e["Solver"]["Trajectory Size"] = 1;
  e["Solver"]["Optimization Steps Per Trajectory"] = 1;

  e["Solver"]["Random Action Probability"]["Initial Value"] = 0.5;
  e["Solver"]["Random Action Probability"]["Target Value"] = 0.05;
  e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.05;

  ////// Defining the configuration of replay memory

  e["Solver"]["Experience Replay"]["Start Size"] =   100;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 100000;

  //// Defining Critic Configuration

  e["Solver"]["Critic"]["Optimizer"]["Type"] = "Optimizer/Adam";
  e["Solver"]["Critic"]["Optimizer"]["Eta"] = 0.001;
  e["Solver"]["Critic"]["Discount Factor"] = 0.99;
  e["Solver"]["Critic"]["Mini Batch Size"] = 64;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Node Count"] = 10;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";

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

  //// Defining Policy Configuration

  e["Solver"]["Policy"]["Optimizer"]["Type"] = "Optimizer/Adam";
  e["Solver"]["Policy"]["Optimizer"]["Eta"] = 0.001;
  e["Solver"]["Policy"]["Mini Batch Size"] = 16;
  e["Solver"]["Policy"]["Adoption Rate"] = 0.50;

  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Node Count"] = 5;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Node Count"] = 2;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Logistic";

  e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = { 1.0, 0.25 };

  ////// Defining Termination Criteria

  e["Solver"]["Training Reward Threshold"] = 300;
  e["Solver"]["Policy Testing Episodes"] = 20;
  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 450;

  ////// Setting file output configuration

  e["File Output"]["Frequency"] = 10000;
  //e["Console Output"]["Verbosity"] = "Silent"

  ////// Running Experiment

  auto k = korali::Engine();
  k.run(e);
}

