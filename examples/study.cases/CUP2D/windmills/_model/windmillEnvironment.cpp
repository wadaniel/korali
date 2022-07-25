#include "windmillEnvironment.hpp"
#include "configs.hpp"
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
  std::cerr<<"Entering runEnvironment function"<<std::endl; //------------------------------------------
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

  std::cerr<<"Before retrieving tasks"<<std::endl; //------------------------------------------

  // retrieving task alpha and task reward and state
  int task_alpha = atoi(_argv[_argc-5]);

  int task_reward = atoi(_argv[_argc-7]);

  int task_state = atoi(_argv[_argc-9]);

  std::string argumentString = OPTIONS + OBJECTS; // this is what we feed to the simulation from configs.hpp
  std::cerr<<argumentString<<std::endl;

  std::stringstream ss(argumentString);
  std::string item;
  std::vector<std::string> arguments;
  while ( std::getline(ss, item, ' ') )
    arguments.push_back(item);

  // Create argc / argv to pass to CUP
  std::vector<char*> argv;
  for (const auto& arg : arguments)
    argv.push_back((char*)arg.data());
  argv.push_back(nullptr);

  std::cerr<<"Before new Simulation"<<std::endl; //------------------------------------------

  // Creating simulation environment
  Simulation *_environment = new Simulation(argv.size()-1, argv.data(), comm);
  _environment->init();

  // Obtaining agents, 2 windmills 
  std::vector<std::shared_ptr<Shape>> shapes = _environment->getShapes();
  Windmill* agent1 = dynamic_cast<Windmill*>(shapes[0].get());
  Windmill* agent2 = dynamic_cast<Windmill*>(shapes[1].get());

  std::cerr<<"After new Simulation"<<std::endl; //------------------------------------------

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


  // now we separate possibilities according to the task at hand

  //-------------------------------------------------------------------------- state
  std::vector<double> state;
  double omega1(0.0);
  double omega2(0.0);


  switch(task_state){
    case 1:
    {
      omega1 = agent1->getAngularVelocity(); // angular velocity of fan 1
      omega2 = agent2->getAngularVelocity(); // angular velocity of fan 2
      state.push_back(omega1); state.push_back(omega2);
      printf("The state is only the angular velocities. \n");
      break;
    }
    case 2:
    {
      state = agent1->vel_profile();
      printf("The state is only the velocity profile. \n");
      break;
    }
    case 3:
    {
      state = agent1->vel_profile();
      omega1 = agent1->getAngularVelocity(); // angular velocity of fan 1
      omega2 = agent2->getAngularVelocity(); // angular velocity of fan 2
      state.push_back(omega1); state.push_back(omega2);
      printf("The state is both the angular velocities and the velocity profile. \n");
      break;
    }
  }

  double num_policy = choosePolicy(task_alpha);
  if (task_alpha == 11) state.push_back(num_policy); //append policy number to the state;

  // print out the state
  printf("[Korali] Initial State: [ %.3f", state[0]);
  for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
  printf("]\n");

  s["State"] = state;

  // load the target profile
  // copy the contents of the avgprofiles.dat file into vector c++
  int num_profiles = 11;

  std::vector<std::vector<double>> profiles(num_profiles, std::vector<double> (32, 0.0)); // initialize vector of size numsteps x 33
  std::vector<std::vector<double>> sigmas(num_profiles, std::vector<double> (32, 0.0));
  
  std::vector<double> target_profile(32, 0.0);
  std::vector<double> sigma_profile(32, 0.0);

  int index = int((num_policy + 1) * 5);
  std::cerr<<"index : "<< index <<std::endl;

  // load the data with rank 0
  if (rank == 0)
  {
    std::ifstream myfile;
    myfile.open("../../avgprofiles.dat", ios::in);

    if (myfile.is_open()){
      std::cerr<<"File is open"<<std::endl;
      std::cerr<<"Succesfully opened the file"<<std::endl;

    } else{
      std::cerr<<"File is closed"<<std::endl;
      std::cerr<<"Failed to open the file"<<std::endl;
    }

    std::string line;
    int i = 0;
    // std::cerr<<"Before while "<< i <<std::endl;
    while (std::getline(myfile, line))
    {
      //std::cerr<<"line "<< i <<std::endl;
      std::istringstream data_line(line);
      int j = 0;
      for (j=0; j < 32; ++j)
      {
        if (data_line >> profiles[i][j])
        {

        } else{
          std::cerr<<"Failed to read number"<<std::endl;
        }
      }
      i += 1;
    }
    myfile.close();

    target_profile = profiles[index];
  }

  // broadcast the target_profile to everyone
  MPI_Bcast(&target_profile.front(), 32, MPI_DOUBLE, 0, comm);
  std::cerr<<"After first broadcast"<<std::endl;
  std::cerr<<target_profile[0]<<" "<<target_profile[1]<<std::endl;

  if (task_reward == 3)
  {
    // load the data with rank 0
    if (rank == 0)
    {
      std::ifstream myfile;
      myfile.open("../../stdprofiles.dat", ios::in);

      if (myfile.is_open()){
        std::cerr<<"File is open"<<std::endl;
        std::cerr<<"Succesfully opened the file"<<std::endl;

      } else{
        std::cerr<<"File is closed"<<std::endl;
        std::cerr<<"Failed to open the file"<<std::endl;
      }

      std::string line;
      int i = 0;
      std::cerr<<"Before while "<< i <<std::endl;
      while (std::getline(myfile, line))
      {
        std::cerr<<"line "<< i <<std::endl;
        std::istringstream data_line(line);
        int j = 0;
        for (j=0; j < 32; ++j)
        {
          if (data_line >> sigmas[i][j])
          {

          } else{
            std::cerr<<"Failed to read number"<<std::endl;
          }
        }
        i += 1;
      }
      myfile.close();

      sigma_profile = sigmas[index];
    }

    // broadcast the target_profile to everyone
    MPI_Bcast(&sigma_profile.front(), 32, MPI_DOUBLE, 0, comm);
  }

  std::vector<double> action(2, 0.0); 

  std::vector<double> profile_t_1 = vector<double>(32, 0.0);
  std::vector<double> sum_profile_t_1 = vector<double>(32, 0.0);
  std::vector<double> profile_t_ = vector<double>(32, 0.0);
  std::vector<double> sum_profile_t_ = vector<double>(32, 0.0);

  std::vector<double> avg_profile_t_1 = vector<double>(32, 0.0);
  std::vector<double> avg_profile_t_ = vector<double>(32, 0.0);
  

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 4000; // 200s of simulations
  double time_step = 0.05;

  double reward = 0;

  bool done = false; // is true if |omega| is superior 10

  // RL loop
  std::cerr<<"Before entering RL Loop"<<std::endl;
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
    //std::cerr<<"Before action Bcast"<<std::endl;
    // broadcast the action to the different processes
    MPI_Bcast(&action[0], 2, MPI_DOUBLE, 0, comm );
    //std::cerr<<"After action Bcast"<<std::endl;

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
    //std::cerr<<"After CUP2D sim steps"<<std::endl;

    profile_t_ = agent1->vel_profile(); // vector of size 32, has the velocity profile values

    omega1 = agent1->getAngularVelocity(); // angular velocity of fan 1
    omega2 = agent2->getAngularVelocity(); // angular velocity of fan 2

    //-------------------------------------------------------------------------- state
    switch(task_state){
      case 1:
      {
        state = {omega1, omega2};
        break;
      }
      case 2:
      {
        state = profile_t_;
        printf("The state is only the velocity profile. \n");
        break;
      }
      case 3:
      {
        state = profile_t_;
        state.push_back(omega1); state.push_back(omega2);
        printf("The state is both the angular velocities and the velocity profile. \n");
        break;
      }
    }

    //std::cerr<<"After state is obtained"<<std::endl;

    if (task_alpha == 11) state.push_back(num_policy); //append policy number to the state;

    // check if angular velocities are over the threshold, true if either of the angular velocities is more than 10
    if (omega1*omega1 > 100 || omega2*omega2 > 100) done = true;

    // must time average the profiles as well before passing them to reward fct
    for(int i(0); i < 32; ++i)
    {
      sum_profile_t_1[i] = sum_profile_t_1[i] + profile_t_1[i];
      if (curStep == 0)
      {
        // the average previous profile is zero
      } else {
        avg_profile_t_1[i] = sum_profile_t_1[i] * time_step / (t-time_step);
      }
      
      sum_profile_t_[i] = sum_profile_t_[i] + profile_t_[i];
      avg_profile_t_[i] = sum_profile_t_[i] * time_step / t;
    }

    //-------------------------------------------------------------------------- reward

    reward = 0; // reset the reward to zero

    double pen_reward = 0.0;

    switch(task_reward){
      case 1:
      {
        // squared difference between deviation from the mean at time t and time t-1, normalized by the target profile
        for (size_t i(0); i < 32; ++i)
        {
          reward -= ((avg_profile_t_[i] - target_profile[i])/target_profile[i]) * ((avg_profile_t_[i] - target_profile[i])/target_profile[i]);
          reward += ((avg_profile_t_1[i] - target_profile[i])/target_profile[i]) * ((avg_profile_t_1[i] - target_profile[i])/target_profile[i]);
        }
        pen_reward = -1000.0;
        break;
      }
      case 2:
      {
        // squared difference between deviation from the mean at time t and time t-1, non-normalized
        for (size_t i(0); i < 32; ++i)
        {
          reward -= (avg_profile_t_[i] - target_profile[i]) * (avg_profile_t_[i] - target_profile[i]);
          reward += (avg_profile_t_1[i] - target_profile[i]) * (avg_profile_t_1[i] - target_profile[i]);
        }
        pen_reward = -200.0;
        break;
      }
      case 3:
      {
        // log-likelihood for hypothetical normal distribution
        for (size_t i(0); i < 32; ++i)
        {
          reward -= (profile_t_[i] - target_profile[i]) * (profile_t_[i] - target_profile[i]) / (2*sigma_profile[i]*sigma_profile[i]);
          reward -= 0.5 * std::log(2 * M_PI * sigma_profile[i] * sigma_profile[i]);
        }
        pen_reward = -400000.0;
        break;
      }
      case 4:
      {
        // no reward is given, goal is to teach agent to not have angular velocity faster than 10
        reward = 2.5e-4;
        pen_reward = -1;
      }
      case 5:
      {
        // difference between current profile and target, non-normalized
        
      }
    }

    // Storing reward
    agent1->printRewards(done ? pen_reward : reward);

    s["Reward"] = done ? pen_reward : reward;
    
    // storing the new profile in the old profile
    profile_t_1 = profile_t_;
    

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

  std::cerr<<"After RL loop"<<std::endl;
  // Setting finalization status
  s["Termination"] = done ? "Terminal" : "Truncated";

  // Flush CUP logger
  logger.flush();

  // Closing log file
  if( rank == 0 ) fclose(logFile);

  std::cerr<<"After closing log file"<<std::endl;

  // delete simulation class
  delete _environment;

  std::cerr<<"After deleting environment"<<std::endl;

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

double choosePolicy(double value)
{
  // returns values between -1 and 1, in steps of 0.2
  double val = value;

  if (value == 11)
  {
    std::uniform_real_distribution<double> dis(-11, 11);
    val = std::floor(dis(_randomGenerator));
    if (std::fmod(val, 2) != 0)
    {
      val += 1;
    }
    val /= 10.0;
  }
  else
  {
    val = value/5.0 - 1; // this is in the range -1, 1
  }
  
  return val;
}

