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

  //std::cerr<<"Runfile : Before task parameters"<<std::endl;

  // retrieving number of ranks, task rnn sequence length, task alpha and task reward and state
  int nRanks = atoi(argv[argc-1]);

  int task_seqlength = atoi(argv[argc-3]);

  int task_alpha = atoi(argv[argc-5]);

  int task_reward = atoi(argv[argc-7]);

  int task_state = atoi(argv[argc-9]);


  std::cerr<< "sequence length : "<<task_seqlength<<std::endl;
  std::cerr<< "alpha : "<<task_alpha<<std::endl;
  std::cerr<< "reward : "<<task_reward<<std::endl;
  std::cerr<< "state : "<<task_state<<std::endl;

  // Storing parameters
  _argc = argc;
  _argv = argv;

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


  //std::cerr<<"Runfile : Before state definition"<<std::endl;
  // states
  int num_states = 0;

  switch(task_state){
    case 1:
    {
      num_states = 2;
      // state is the angular velocities
      e["Variables"][0]["Name"] = std::string("Omega 1");
      e["Variables"][0]["Type"] = "State";
      e["Variables"][1]["Name"] = std::string("Omega 2");
      e["Variables"][1]["Type"] = "State";
      break;
    }
    case 2:
    {
      num_states = 32;
      // state is the velocity profile
      for (size_t i = 0; i < 32; i++)
      {
        e["Variables"][i]["Name"] = std::string("Velocity ") + std::to_string(i+1);
        e["Variables"][i]["Type"] = "State";
      }
      break;
    }
    case 3:
    {
      // state is angular velocities and velocity profile
      num_states = 34;
      // state is the velocity profile
      for (size_t i = 0; i < 32; i++)
      {
        e["Variables"][i]["Name"] = std::string("Velocity ") + std::to_string(i+1);
        e["Variables"][i]["Type"] = "State";
      }

      e["Variables"][32]["Name"] = std::string("Omega 1");
      e["Variables"][32]["Type"] = "State";
      e["Variables"][33]["Name"] = std::string("Omega 2");
      e["Variables"][33]["Type"] = "State";
      break;
    }
  }

  if (task_alpha == 11){
    e["Variables"][num_states]["Name"] = std::string("Policy number");
    e["Variables"][num_states]["Type"] = "State";
    num_states += 1;
  }

  // actions
  double max_angular_acceleration = 15;
  double exploration_noise = 4;

  e["Variables"][num_states]["Name"] = "Angular acceleration 1";
  e["Variables"][num_states]["Type"] = "Action";
  e["Variables"][num_states]["Lower Bound"] = -max_angular_acceleration;
  e["Variables"][num_states]["Upper Bound"] = +max_angular_acceleration;
  e["Variables"][num_states]["Initial Exploration Noise"] = exploration_noise;

  e["Variables"][num_states + 1]["Name"] = "Angular acceleration 2";
  e["Variables"][num_states + 1]["Type"] = "Action";
  e["Variables"][num_states + 1]["Lower Bound"] = -max_angular_acceleration;
  e["Variables"][num_states + 1]["Upper Bound"] = +max_angular_acceleration;
  e["Variables"][num_states + 1]["Initial Exploration Noise"] = exploration_noise;

  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Concurrent Environments"] = N;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 1.0; // discount factor set to 1
  e["Solver"]["Mini Batch"]["Size"] =  128;


  //--------------------------------------------------------------------------------------------------------//
  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

  //// Defining Policy distribution and scaling parameters
  e["Solver"]["Policy"]["Distribution"] = "Clipped Normal";
  e["Solver"]["State Rescaling"]["Enabled"] = false;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = false;

  // Configuring the neural network and its hidden layers
  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";
  
  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  // recurrent network
  e["Solver"]["Time Sequence Length"] = task_seqlength; // length of time sequence, corresponding to number of time steps

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Recurrent/GRU";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Depth"] = 1;
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Linear";
  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"] = 128;

  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

  // feedforward network
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  // e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  // e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  // e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  // e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  // e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";
  
  ////// Defining Termination Criteria
  e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e7;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Use Multiple Files"] = false;
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

  std::cout<<"Before Run"<<std::endl;

  // run korali
  k.run(e);
}