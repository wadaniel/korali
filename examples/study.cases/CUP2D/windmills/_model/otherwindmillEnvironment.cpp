#include "windmillEnvironment.hpp"
#include "Operators/Helpers.h"
#include "Definitions.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

// 2 windmills with variable angular acceleration applied to them
// mpi version of the code
void runEnvironment(korali::Sample &s)
{
  // Get MPI communicator
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();

  // Get rank and size of subcommunicator
  int rank, size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  // Get rank in world
  int rankGlobal;
  MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);

  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%03u", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rankGlobal/size);
  if( rank == 0 )
  if( not std::filesystem::exists(resDir) )
  if( not std::filesystem::create_directories(resDir) )
  {
    fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
    exit(-1);
  };

  // Redirecting all output to the log file
  FILE * logFile;
  if( rank == 0 ) {
    char logFilePath[128];
    sprintf(logFilePath, "%s/log.txt", resDir);
    logFile = freopen(logFilePath, "w", stdout);
    if (logFile == NULL)
    {
      printf("[Korali] Error creating log file: %s.\n", logFilePath);
      exit(-1);
    }
  }
  // Make sure folder / logfile is created before switching path
  MPI_Barrier(comm);

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv, comm);
  _environment->init();

  // Obtaining agents, 2 windmills 
  std::vector<std::shared_ptr<Shape>> shapes = _environment->getShapes();
  Windmill* agent1 = dynamic_cast<Windmill*>(shapes[0].get());
  Windmill* agent2 = dynamic_cast<Windmill*>(shapes[1].get());

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting initial conditions, two fans are initialized with an angle of zero degrees
  setInitialConditions(agent1, 0.0, false);
  setInitialConditions(agent2, 0.0, false);

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();


  // the state of the environment is a vector of size 35:
  // the first 32 components are the velocity profile at the target
  // the next 2 componenets are the angular velocities of the windmills
  // the final component is a number t between -1 and 1, steps of 0.2
  // it tells the network which policy we want to obtain

  // Setting initial state [Careful, state function needs to be called by all ranks!]
  // the state function returns the entire state already, done in CUP
  std::vector<double> state = agent1->vel_profile(); // vector of size 32, has the velocity profile values
  double omega1 = agent1->getAngularVelocity(); // angular velocity of fan 1
  double omega2 = agent2->getAngularVelocity(); // angular velocity of fan 2
  // we choose one constant policy
  double num_policy = choosePolicy(0, false); // number to give to the agent to decide which policy to follow, between -1 and 1
  state.push_back(omega1); state.push_back(omega2); state.push_back(num_policy);// then append them all to the state

  printf("[Korali] Initial State: [ %.3f", state[0]);
  for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
  printf("]\n");


  s["State"] = state;

  // load the target profile
  // copy the contents of the avgprofiles.dat file into vector c++

  // std::vector<std::vector<double>> profiles(4000, std::vector<double> (32, 0.0)); // initialize vector of size numsteps x 32
  // std::vector<std::vector<double>> stds(4000, std::vector<double> (32, 0.0)); // initialize vector of size numsteps x 32

  // // load the data with rank 0
  // if (rank == 0)
  // {
  //   int index = int((num_policy + 1) * 5);
  //   std::cout<<"index : "<< index <<std::endl;

  //   std::string file_name = "../../avgprofile_" + std::to_string(index) + ".dat";

  //   std::ifstream myfile;
  //   myfile.open(file_name, ios::in);

  //   if (myfile.is_open()){
  //     std::cout<<"File is open"<<std::endl;
  //     std::cerr<<"Succesfully opened the file"<<std::endl;

  //   } else{
  //     std::cout<<"File is closed"<<std::endl;
  //     std::cerr<<"Failed to open the file"<<std::endl;
  //   }

  //   std::string line;
  //   int i = 0;
  //   std::cout<<"Before while "<< i <<std::endl;
  //   while (std::getline(myfile, line))
  //   {
  //     std::cout<<"line "<< i <<std::endl;
  //     std::istringstream data_line(line);
  //     int j = 0;
  //     for (j=0; j < 32; ++j)
  //     {
  //       if (data_line >> profiles[i][j])
  //       {

  //       } else{
  //         std::cout<<"Failed to read number"<<std::endl;
  //       }
  //     }
  //     i += 1;
  //   }
  //   myfile.close();

  //   // std file
  //   file_name = "../../stdprofile_" + std::to_string(index) + ".dat";

  //   myfile;
  //   myfile.open(file_name, ios::in);

  //   if (myfile.is_open()){
  //     std::cout<<"File is open"<<std::endl;
  //     std::cerr<<"Succesfully opened the file"<<std::endl;

  //   } else{
  //     std::cout<<"File is closed"<<std::endl;
  //     std::cerr<<"Failed to open the file"<<std::endl;
  //   }

  //   i = 0;
  //   std::cout<<"Before while "<< i <<std::endl;
  //   while (std::getline(myfile, line))
  //   {
  //     std::cout<<"line "<< i <<std::endl;
  //     std::istringstream data_line(line);
  //     int j = 0;
  //     for (j=0; j < 32; ++j)
  //     {
  //       if (data_line >> stds[i][j])
  //       {

  //       } else{
  //         std::cout<<"Failed to read number"<<std::endl;
  //       }
  //     }
  //     i += 1;
  //   }
  //   myfile.close();
  // }

  int num_profiles = 11;

  std::vector<std::vector<double>> profiles(num_profiles, std::vector<double> (32, 0.0)); // initialize vector of size numsteps x 33
  std::vector<double> target_profile(32, 0.0);

  // load the data with rank 0
  if (rank == 0)
  {
    std::ifstream myfile;
    myfile.open("../../avgprofiles.dat", ios::in);

    if (myfile.is_open()){
      std::cout<<"File is open"<<std::endl;
      std::cerr<<"Succesfully opened the file"<<std::endl;

    } else{
      std::cout<<"File is closed"<<std::endl;
      std::cerr<<"Failed to open the file"<<std::endl;
    }

    std::string line;
    int i = 0;
    std::cout<<"Before while "<< i <<std::endl;
    while (std::getline(myfile, line))
    {
      std::cout<<"line "<< i <<std::endl;
      std::istringstream data_line(line);
      int j = 0;
      for (j=0; j < 32; ++j)
      {
        if (data_line >> profiles[i][j])
        {

        } else{
          std::cout<<"Failed to read number"<<std::endl;
        }
      }
      i += 1;
    }
    myfile.close();

    int index = int((num_policy + 1) * 5);
    std::cout<<"index : "<< index <<std::endl;
    target_profile = profiles[index];
  }

  // broadcast the target_profile to everyone
  // MPI_Bcast(&target_profile.front(), 128000, MPI_DOUBLE, 0, comm);
  // std::cout<<"After first broadcast"<<std::endl;
  // std::cout<<target_profile[0]<<" "<<target_profile[1]<<std::endl;

  // compute the norm of the target_profile
  // double norm_prof = 0;

  // for (auto e : target_profile)
  // {
  //   norm_prof += e * e;
  // }
  // norm_prof = std::sqrt(norm_prof);

  std::vector<double> action(2, 0.0); 

  std::vector<double> profile_t_ = vector<double>(32, 0.0);
  std::vector<double> sum_profile_t_ = vector<double>(32, 0.0);
  std::vector<double> avg_profile_t_ = vector<double>(32, 0.0);
  

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 4000; // 200s of simulations

  double reward = 0;

  bool done = false; // is true if |omega| is superior 10

  // RL loop
  while (curStep < maxSteps && done == false)
  {
    // Only rank 0 communicates with Korali
    if (rank == 0)
    {
      // Getting new action
      s.update();

      // Reading new action
      auto actionJSON = s["Action"];
      action = actionJSON.get<std::vector<double>>();
    }

    // broadcast the action to the different processes
    MPI_Bcast(&action[0], 2, MPI_DOUBLE, 0, comm );

    // Setting action for 
    agent1->act( action[0] );
    agent2->act( action[1] );

    // Run the simulation until next action is required
    tNextAct += 0.05 ;
    while ( t < tNextAct )
    {
      // Calculate simulation timestep
      const double dt = std::min(_environment->calcMaxTimestep(), 0.05);
      t += dt;

      // Advance simulation
      _environment->advance(dt);
    }

    profile_t_ = agent1->vel_profile(); // vector of size 32, has the velocity profile values
    state = profile_t_;
    omega1 = agent1->getAngularVelocity(); // angular velocity of fan 1
    omega2 = agent2->getAngularVelocity(); // angular velocity of fan 2
    state.push_back(omega1); state.push_back(omega2); state.push_back(num_policy);// then append them all to the state

    // check if angular velocities are over the threshold, true if either of the angular velocities is more than 12
    done = (omega1*omega1 > 144 || omega2*omega2 > 144) ? true : false;

    // reset reward
    reward = 0;

    // reward 1 : deviation from mean profile
    for(int i(0); i < 32; ++i)
    {
      reward -=  ((profile_t_[i] - target_profile[i]) / target_profile[i]) * ((profile_t_[i] - target_profile[i]) / target_profile[i]);
    }

    // must time average the profiles as well before passing them to reward fct
    // for(int i(0); i < 32; ++i)
    // {
    //   sum_profile_t_[i] = sum_profile_t_[i] + profile_t_[i];
    //   avg_profile_t_[i] = sum_profile_t_[i] / t;

    //   //reward += -(avg_profile_t_[i] - profiles[curStep][i]) * (avg_profile_t_[i] - profiles[curStep][i]) / stds[curStep][i];
    //   //reward += -std::abs(avg_profile_t_[i] - profiles[curStep][i]) / std::sqrt(stds[curStep][i]);
    //   reward += - std::sqrt((avg_profile_t_[i] - profiles[curStep][i]) * (avg_profile_t_[i] - profiles[curStep][i]) / profiles[curStep][i]);
    // }

    // Storing reward
    double pen_reward = -40000.0;
    agent1->printRewards(done ? pen_reward : reward);

    s["Reward"] = done ? pen_reward : reward;

    

    // Storing new state
    s["State"] = state;

    if( rank == 0 ) {
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      printf("[Korali] State: [ %.3f", state[0]);
      for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
      printf("]\n");
      printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
      printf("[Korali] Reward: %.3f\n", reward);
      printf("[Korali] -------------------------------------------------------\n");
    }

    // Advancing to next step
    curStep++;
  }

  // Setting finalization status
  s["Termination"] = done ? "Terminal" : "Truncated";

  // Flush CUP logger
  logger.flush();

  // Closing log file
  if( rank == 0 ) fclose(logFile);

  // delete simulation class
  delete _environment;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

}


// set initial conditions of the agent
void setInitialConditions(Windmill* agent, double init_angle, bool randomized)
{
  // Intial fixed condition of angle and angular velocity
  double angle = init_angle;

  // set random beginning angle
  if(randomized)
  {
    // windmills have 3 axis of symmetry, meaning if we rotate it by 120°,we have the same setup
    std::uniform_real_distribution<double> dis(0, 2*(M_PI/3));
    angle = dis(_randomGenerator);
  }
  
  printf("[Korali] Initial Conditions:\n");
  printf("[Korali] orientation: %f\n", angle);

  agent->setOrientation(angle);
}

double choosePolicy(double value, bool randomized)
{
  // returns values between -1 and 1, in steps of 0.2
  double val = value;

  if (randomized)
  {
    std::uniform_real_distribution<double> dis(-11, 11);
    val = std::floor(dis(_randomGenerator));
    if (std::fmod(val, 2) != 0)
    {
      val += 1;
    }
  }
  val /= 10.0;
  return val;
}