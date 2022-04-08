// Select which environment to use
#include "_model/windmillEnvironment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  // Gathering actual arguments from MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // Storing parameters
  _argc = argc;
  _argv = argv;

  // retrieving number of ranks
  int nRanks = atoi(argv[argc-1]);

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine
  N = (int)(N/nRanks); // divide by number of ranks per worker

  // Set results path
  std::string trainingResultsPath = "_trainingResults/";
  //std::string testingResultsPath = "../_results_windmill_testing/";
  
  // Creating Korali experiment
  auto e = korali::Experiment();

  // Check if there is log files to continue training
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  // Configuring problem (for test eliminate after)
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;

  // Adding custom setting to run the environment without dumping the state files during training
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;
  
  const size_t profileStates = 32;
  for (size_t i = 0; i < profileStates; i++)
  {
    e["Variables"][i]["Name"] = std::string("Velocity ") + std::to_string(i+1);
    e["Variables"][i]["Type"] = "State";
  }

  e["Variables"][32]["Name"] = std::string("Omega 1");
  e["Variables"][32]["Type"] = "State";
  e["Variables"][33]["Name"] = std::string("Omega 2");
  e["Variables"][33]["Type"] = "State";
  e["Variables"][34]["Name"] = std::string("Policy number");
  e["Variables"][34]["Type"] = "State";

  double max_angular_acceleration = 15;
  double exploration_noise = 12;

  e["Variables"][35]["Name"] = "Angular acceleration 1";
  e["Variables"][35]["Type"] = "Action";
  e["Variables"][35]["Lower Bound"] = -max_angular_acceleration;
  e["Variables"][35]["Upper Bound"] = +max_angular_acceleration;
  e["Variables"][35]["Initial Exploration Noise"] = exploration_noise;

  e["Variables"][36]["Name"] = "Angular acceleration 2";
  e["Variables"][36]["Type"] = "Action";
  e["Variables"][36]["Lower Bound"] = -max_angular_acceleration;
  e["Variables"][36]["Upper Bound"] = +max_angular_acceleration;
  e["Variables"][36]["Initial Exploration Noise"] = exploration_noise;

  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Concurrent Environments"] = N;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.95; // used to be 0.95
  e["Solver"]["Mini Batch"]["Size"] =  128;
  
  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

  //// Defining Policy distribution and scaling parameters
  e["Solver"]["Policy"]["Distribution"] = "Clipped Normal";
  e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;

  // Configuring the neural network and its hidden layers
  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";
  
  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";
  
  ////// Defining Termination Criteria
  e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e7;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = trainingResultsPath;

  ////// Running Experiment
  auto k = korali::Engine();

  // Configuring profiler output
  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

  // set conduit and MPI communicator
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Ranks Per Worker"] = nRanks;
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  // run korali
  k.run(e);
}

///////////////////////////////////////////////////////////

  // const size_t numStates = 1024;
  // for (size_t i = 0; i < numStates; i++)
  // {
  //   e["Variables"][i]["Name"] = std::string("Velocity ") + std::to_string(i);
  //   e["Variables"][i]["Type"] = "State";
  // }
  
  // //action is now setting the torque of the fans
  // double max_torque = 5e-4;
  // for(size_t j=numStates; j < numStates + 2; ++j){
  //   e["Variables"][j]["Name"] = "Torque " + std::to_string(j-numStates+1);
  //   e["Variables"][j]["Type"] = "Action";
  //   e["Variables"][j]["Lower Bound"] = -max_torque;
  //   e["Variables"][j]["Upper Bound"] = +max_torque;
  //   e["Variables"][j]["Initial Exploration Noise"] = 0.5;
  // }

  /////////////////////////////////////////////////////////////////////////////////

  // neural network used when state was angle and angular velocity, about 8292 parameters
  /*
  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";
  
  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  // Convolutional Layer with ReLU activation function [1x32x32] -> [4x28x28]
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"]      = 32;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"]       = 32;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"]     = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"]      = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"]   = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"]   = 4*28*28;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/ReLU";

  // Pooling Layer [4x28x28] -> [4x14x14]
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Pooling";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"]          = "Exclusive Average";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Height"]      = 28;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Width"]       = 28;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Height"]     = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Width"]      = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Vertical Stride"]   = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Horizontal Stride"] = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"]   = 4*14*14;

  // Convolutional Layer with tanh activation function [4x14x14] -> [16x10x10]
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"]      = 14;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"]       = 14;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"]     = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"]      = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"]   = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"]   = 16*10*10;

  e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/ReLU";

  // Pooling Layer [16x10x10] -> [16x5x5]
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Pooling";
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"]          = "Exclusive Average";
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Height"]      = 10;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Width"]       = 10;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Height"]     = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Width"]      = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Vertical Stride"]   = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Horizontal Stride"] = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Output Channels"]   = 16*5*5;

  // Convolutional Fully Connected Output Layer [16x5x5] -> [64x2x2]
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution";
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"]      = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"]       = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"]     = 4;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"]      = 4;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"]   = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"]   = 64*2*2;

  e["Solver"]["Neural Network"]["Hidden Layers"][7]["Type"] = "Layer/Activation"; // there was a 4 here before, so basically did nothing after
  e["Solver"]["Neural Network"]["Hidden Layers"][7]["Function"] = "Elementwise/ReLU"; // the previous conv layer, vel_oui_relu_30s when added

  // Convolutional Fully Connected Output Layer [64x2x2] -> [64x1x1]
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Type"] = "Layer/Pooling";
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Function"]          = "Exclusive Average";
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Height"]      = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Width"]       = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Height"]     = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Width"]      = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Vertical Stride"]   = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Horizontal Stride"] = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][8]["Output Channels"]   = 64*1*1;


  // addition to original net

  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/ReLU";

  */
  /////////////////////////////////////////////////////////////////////////////////

  // neural network used when state was angle and angular velocity, about 17664 parameters
  
  

  /////////////////////////////////////////////////////////////////////////////////

