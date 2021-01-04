#include "_environment/environment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  auto e = korali::Experiment();

  //// Defining Experiment

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 1.0;
  e["Problem"]["Policy Testing Episodes"] = 20;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  e["Variables"][0]["Name"] = "Mean u at wall";
  e["Variables"][0]["Type"] = "State";

  e["Variables"][1]["Name"] = "Mean u off wall";
  e["Variables"][1]["Type"] = "State";

  e["Variables"][2]["Name"] = "Mean du/dy at wall";
  e["Variables"][2]["Type"] = "State";

  e["Variables"][3]["Name"] = "Mean du/dy off wall";
  e["Variables"][3]["Type"] = "State";

  e["Variables"][4]["Name"] = "Mean w at wall";
  e["Variables"][4]["Type"] = "State";

  e["Variables"][5]["Name"] = "Mean w off wall";
  e["Variables"][5]["Type"] = "State";

  e["Variables"][6]["Name"] = "Boundary Condition (x)";
  e["Variables"][6]["Type"] = "Action";
  e["Variables"][6]["Lower Bound"] = 0.0;
  e["Variables"][6]["Upper Bound"] = 0.01;
  e["Variables"][6]["Exploration Sigma"]["Initial"] = 0.0005;
  e["Variables"][6]["Exploration Sigma"]["Final"] = 0.0005;
  e["Variables"][6]["Exploration Sigma"]["Annealing Rate"] = 0.0;

  /// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent / Continuous / GFPT";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Time Sequence Length"] = 1;
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Cache Persistence"] = 10;
  e["Solver"]["Discount Factor"] = 0.99;

  /// Defining the configuration of replay memory

  e["Solver"]["Mini Batch Size"] = 32;
  e["Solver"]["Mini Batch Strategy"] = "Uniform";
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 32768;
  e["Solver"]["Experience Replay"]["Serialization"]["Frequency"] = 1;

  /// Defining Critic and Policy Configuration

  e["Solver"]["Learning Rate"] = 0.01;
  e["Solver"]["Policy"]["Learning Rate Scale"] = 1.0;
  e["Solver"]["Critic"]["Advantage Function Population"] = 12;
  e["Solver"]["Policy"]["Target Accuracy"] = 0.001;
  e["Solver"]["Policy"]["Optimization Candidates"] = 12;

  /// Configuring the neural network and its hidden layers

  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32;

  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria

  e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 1.3;

  ////// Setting file output configuration

  e["File Output"]["Enabled"] = false;

  ////// Running Experiment

  auto k = korali::Engine();
  k.run(e);
}
