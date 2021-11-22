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

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

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
  // e["Problem"]["Training Reward Threshold"] = 8.0;
  // e["Problem"]["Policy Testing Episodes"] = 5;

  // Adding custom setting to run the environment without dumping the state files during training
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;


  ////////////////////////////////////////////////////////// 
  // const size_t numStates = 4;
  // e["Variables"][0]["Name"] = std::string("Angle ") + std::to_string(0);
  // e["Variables"][1]["Name"] = std::string("Omega ") + std::to_string(0);
  // e["Variables"][2]["Name"] = std::string("Angle ") + std::to_string(1);
  // e["Variables"][3]["Name"] = std::string("Omega ") + std::to_string(1);

  // for (size_t i = 0; i < 4; i++) e["Variables"][i]["Type"] = "State";

  ///////////////////////////////////////////////////////////

  const size_t numStates = 576;
  for (size_t i = 0; i < numStates; i++)
  {
    e["Variables"][i]["Name"] = std::string("Vorticity ") + std::to_string(i);
    e["Variables"][i]["Type"] = "State";
  }

  /////////////////////////////////////////////////////////

  // double max_torque = 2.5e-4;
  // action is now setting the velocity of the fans
  double max_omega = 15;
  for(size_t j=numStates; j < numStates + 2; ++j){
    e["Variables"][j]["Name"] = "Omega " + std::to_string(j-numStates+1);
    e["Variables"][j]["Type"] = "Action";
    e["Variables"][j]["Lower Bound"] = -max_omega;
    e["Variables"][j]["Upper Bound"] = +max_omega;
    e["Variables"][j]["Initial Exploration Noise"] = 15;
  }

  // std::cout<<"Before state 2"<<std::endl;

  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Concurrent Environments"] = N;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.99; // used to be 0.95
  e["Solver"]["Mini Batch"]["Size"] =  128;
  //e["Solver"]["Policy"]["Distribution"] = "Normal";
  e["Solver"]["Policy"]["Distribution"] = "Clipped Normal";

  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;


  //// Defining Policy distribution and scaling parameters
  e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = false; // this was true

  /////////////////////////////////////////////////////////////////////////////////

  // neural network used when state was angle and angular velocity, about 8292 parameters

  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";
  
  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  // Convolutional Layer with ReLU activation function [1x24x24] -> [4x20x20]
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"]      = 24;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"]       = 24;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"]     = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"]      = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"]   = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"]   = 4*20*20;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/ReLU";

  // Pooling Layer [4x20x20] -> [4x10x10]
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Pooling";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"]          = "Exclusive Average";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Height"]      = 20;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Width"]       = 20;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Height"]     = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Width"]      = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Vertical Stride"]   = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Horizontal Stride"] = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"]   = 4*10*10;

  // Convolutional Layer with tanh activation function [4x10x10] -> [12x6x6]
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"]      = 10;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"]       = 10;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"]     = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"]      = 5;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"]   = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"]   = 12*6*6;

  e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/ReLU";

  // Pooling Layer [12x6x6] -> [12x3x3]
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Pooling";
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"]          = "Exclusive Average";
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Height"]      = 6;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Width"]       = 6;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Height"]     = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Width"]      = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Vertical Stride"]   = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Horizontal Stride"] = 2;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][5]["Output Channels"]   = 12*3*3;

  // Convolutional Fully Connected Output Layer [12x3x3] -> [64x1x1]
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution";
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"]      = 3;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"]       = 3;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"]     = 3;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"]      = 3;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"]   = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"]      = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"]     = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"]       = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"]    = 0;
  e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"]   = 64*1*1;

  e["Solver"]["Neural Network"]["Hidden Layers"][7]["Type"] = "Layer/Activation"; // there was a 4 here before, so basically did nothing after
  e["Solver"]["Neural Network"]["Hidden Layers"][7]["Function"] = "Elementwise/ReLU"; // the previous conv layer, vel_oui_relu_30s when added



  // addition to original net

  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/ReLU";


  /////////////////////////////////////////////////////////////////////////////////

  // neural network used when state was angle and angular velocity, about 17664 parameters
  
  /// Configuring the neural network and its hidden layers
  // e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  // e["Solver"]["Neural Network"]["Optimizer"] = "Adam";
  
  // e["Solver"]["L2 Regularization"]["Enabled"] = true;
  // e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  // e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  // e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";
  

  /////////////////////////////////////////////////////////////////////////////////

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
  korali::setKoraliMPIComm(MPI_COMM_WORLD);
  // std::cout<<"Before state 4"<<std::endl;

  // run korali
  k.run(e);
}
