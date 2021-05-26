// Select which environment to use
#include "_models/windmillEnvironment/windmillEnvironment.hpp"
#include "korali.hpp"

std::string _resultsPath;

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
  // _argc = argc
  // _argv = argv;

  // this enables to pass the folder as an argument to exe

  _argc = argc-1;

  char *_argv_[argc-1];
  for(int i = 0; i < argc-1; ++i)
  {
    _argv_[i] = argv[i];
  }

  

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Initialize CUP2D
  _environment = new Simulation(_argc, _argv_);
  _environment->init();

  std::string folder = std::string(argv[argc-1]);

  // Set results path
  std::string trainingResultsPath = "_results_windmill_training/" + folder;
  std::string testingResultsPath = "_results_windmill_training/" + folder;
  
  // Creating Korali experiment
  auto e = korali::Experiment();

  // Check if there is log files to continue training
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  auto k = korali::Engine();

  // k["Profiling"]["Detail"] = "Full";
  // k["Profiling"]["Path"] = testingResultsPath + std::string("/profiling.json");
  // k["Profiling"]["Frequency"] = 60;

  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;

  // Adding custom setting to run the environment dumping the state files during testing
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
  e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;
  e["File Output"]["Path"] = testingResultsPath;
  e["Solver"]["Testing"]["Policy"] = e["Solver"]["Best Training Hyperparamters"];
  e["Solver"]["Mode"] = "Testing";
  for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = i;

  k.run(e);
}

// plot policy (state vs action), do this per agent
// action agent took over time and energy consumption it corresponds to, plot mean and std of action and energy consumption
// state over time, mean and std over time (over the 54 diffferent tries)
// policy histogram, how the actions are distributed
// velocity at target over time, mean and std as well over time
// movie of sim
