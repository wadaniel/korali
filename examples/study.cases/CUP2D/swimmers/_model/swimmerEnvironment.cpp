//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment.hpp"
#include "configs.hpp"

#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

// Swimmer following an obstacle
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
  if( s["Mode"] == "Training" )
    sprintf(resDir, "%s/sample%03u", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rankGlobal/size);
  else
    sprintf(resDir, "%s/sample%03lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
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

  // Argument string to inititialize Simulation
  std::string argumentString;

  // Get task from command line argument
  auto task = atoi(_argv[_argc-5]);

  // Get get task/obstacle we want
  if(task == -2 )
  {
    if( s["Mode"] == "Training" )
    {
      // Sample task
      std::uniform_int_distribution<> disT(0, 1);
      task = disT(_randomGenerator);
      // For multitask learning, Korali has to know the task
      s["Environment Id"] = task;
    }
    else
    {
      task = std::floor( sampleId / 10 );
      s["Environment Id"] = task;
    }
  }

  /* Add Obstacle */
  switch(task) {
    case -1 : // SINGLE SWIMMER
    {
      argumentString = "CUP-RL " + OPTIONS;
      break;
    }
    case 0 : // DCYLINDER
    {
      // Only rank 0 samples the radius
      double radius;
      if( s["Mode"] == "Training" )
      {
        if( rank == 0 )
        {
          std::uniform_real_distribution<double> radiusDist(0.03,0.07);
          radius = radiusDist(_randomGenerator);
        }
      }
      else
      {
        radius = 0.03 + (sampleId%10)*(0.07-0.03)/9.0;
      }

      // Broadcast radius to the other ranks
      MPI_Bcast(&radius, 1, MPI_DOUBLE, 0, comm);

      // Set argument string
      argumentString = "CUP-RL " + OPTIONS + " -shapes " + OBJECTShalfDisk + std::to_string(radius);
      break;
    }
    case 1 : // HYDROFOIL
    {
      // Only rank 0 samples the frequency
      double frequency;
      if( s["Mode"] == "Training" )
      {
        if( rank == 0 )
        {
          std::uniform_real_distribution<double> frequencyDist(0.2,0.5);
          frequency = frequencyDist(_randomGenerator);
        }
      }
      else
      {
        frequency = 0.2 + (sampleId%10)*(0.5-0.2)/9.0;
      }

      // Broadcast frequency to the other ranks
      MPI_Bcast(&frequency, 1, MPI_DOUBLE, 0, comm);

      // Set argument string
      argumentString = "CUP-RL " + OPTIONS + " -shapes " + OBJECTSnaca + std::to_string(frequency);
      break;
    }
    case 2 : // STEFANFISH
    {
      // Only rank 0 samples the length
      double length;
      if( s["Mode"] == "Training" )
      {
        if( rank == 0 )
        {
          std::uniform_real_distribution<double> lengthDist(0.15,0.25);
          length = lengthDist(_randomGenerator);
        }
      }
      else
      {
        length = 0.15 + (sampleId%10)*(0.25-0.15)/9.0;
      }
      
      // Broadcast length to the other ranks
      MPI_Bcast(&length, 1, MPI_DOUBLE, 0, comm);

      // Set argument string
      argumentString = "CUP-RL " + OPTIONS + " -shapes " + OBJECTSstefanfish + std::to_string(length);
      break;
    }
    case 3 :
    {
      argumentString = "CUP-RL " + OPTIONS + " -shapes " + OBJECTSwaterturbine;
      break;
    }
  }

  // retreiving number of agents
  int nAgents = atoi(_argv[_argc-3]);

  /* Add Agent(s) */

  // Declare initial data vector
  double initialData[3];

  // Set initial angle
  initialData[0] = 0.0;
  for( int a = 0; a < nAgents; a++ )
  {
    // Set initial position
    std::vector<double> initialPosition{ 0.9, 0.5 };
    if( nAgents > 1)
    {
      initialPosition = initialPositions[a];
    }
    initialData[1] = initialPosition[0];
    initialData[2] = initialPosition[1];

    // During training, add noise to inital configuration of agent
    if ( s["Mode"] == "Training" ) 
    {
      // only rank 0 samples initial data
      if( rank == 0 )
      {
        std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
        std::uniform_real_distribution<double> disX(-0.05, 0.05);
        std::uniform_real_distribution<double> disY(-0.05, 0.05);
        initialData[0] = initialData[0] + disA(_randomGenerator);
        initialData[1] = initialData[1] + disX(_randomGenerator);
        initialData[2] = initialData[2] + disY(_randomGenerator);
      }
      // Broadcast initial data to all ranks
      MPI_Bcast(initialData, 3, MPI_DOUBLE, 0, comm);
    }

    // Append agent to argument string
    argumentString = argumentString + AGENT + AGENTANGLE + std::to_string(initialData[0]) + AGENTPOSX + std::to_string(initialData[1]) + AGENTPOSY + std::to_string(initialData[2]);
  }

  // printf("%s\n",argumentString.c_str());
  // fflush(0);

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

  // Creating simulation environment
  Simulation *_environment = new Simulation(argv.size() - 1, argv.data(), comm);
  _environment->init();

  // Obtaining agents
  std::vector<std::shared_ptr<Shape>> shapes = _environment->getShapes();
  std::vector<StefanFish *> agents(nAgents);
  if( task == -1 )
  {
    agents[0] = dynamic_cast<StefanFish *>(shapes[0].get());
  }
  else 
  {
    for( int i = 1; i<nAgents+1; i++ )
      agents[i-1] = dynamic_cast<StefanFish *>(shapes[i].get());
  }

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting initial state [Careful, state function needs to be called by all ranks!]
  if( nAgents > 1 ) {
    std::vector<std::vector<double>> states(nAgents);
    for( int i = 0; i<nAgents; i++ )
    {
      std::vector<double> initialPosition = initialPositions[i];
      std::vector<double> state = agents[i]->state(initialPosition);

      // assign state/reward to container
      states[i]  = state;
    }
    s["State"] = states;
  }
  else
  {
    std::vector<double> initialPosition{ 0.9, 0.5 };
    s["State"] = agents[0]->state(initialPosition);
  }

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;

  // Careful, hardcoded the number of action(s)!
  std::vector<std::vector<double>> actions(nAgents, std::vector<double>(2));

  // Starting main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    // Only rank 0 communicates with Korali
    if( rank == 0 ) {
      // Getting new action(s)
      s.update();
      auto actionsJson = s["Action"];

      // Setting action for each agent
      for( int i = 0; i<nAgents; i++ )
      {
        std::vector<double> action;
        if( nAgents > 1 )
          action = actionsJson[i].get<std::vector<double>>();
        else
          action = actionsJson.get<std::vector<double>>();
        actions[i] = action;
      }
    }

    // Broadcast and apply action(s) [Careful, hardcoded the number of action(s)!]
    for( int i = 0; i<nAgents; i++ )
    {
      MPI_Bcast( actions[i].data(), 2, MPI_DOUBLE, 0, comm );
      if( actions[i].size() != 2 ) std::cout << "Korali returned the wrong number of actions " << actions[i].size() << "\n";
      agents[i]->act(t, actions[i]);
    }

    // Run the simulation until next action is required
    dtAct = 0.;
    for( int i = 0; i<nAgents; i++ )
    if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
      dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);

      // Check if there was a collision -> termination.
      done = _environment->sim.bCollision;

      // Check termination because leaving margins
      for( int i = 0; i<nAgents; i++ )
        done = ( done || isTerminal(agents[i], nAgents) );
    }

    // Get and store state and reward [Carful, state function needs to be called by all ranks!] 
    if( nAgents > 1 ) {
      std::vector<std::vector<double>> states(nAgents);
      std::vector<double> rewards(nAgents);
      for( int i = 0; i<nAgents; i++ )
      {
        std::vector<double> initialPosition = initialPositions[i];
        std::vector<double> state = agents[i]->state(initialPosition);

        // assign state/reward to container
        states[i]  = state;
        rewards[i] = done ? -10.0 : agents[i]->EffPDefBnd;
      }
      s["State"]  = states;
      s["Reward"] = rewards;
    }
    else {
      std::vector<double> initialPosition{ 0.9, 0.5 };
      s["State"] = agents[0]->state(initialPosition);
      s["Reward"] = done ? -10.0 : agents[0]->EffPDefBnd;
    }

    // Printing Information:
    if( rank == 0 ) {
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      if( nAgents > 1 )
      {
        for( int i = 0; i<nAgents; i++ )
        {
          auto state  = s["State"][i].get<std::vector<float>>();
          auto action = s["Action"][i].get<std::vector<float>>();
          auto reward = s["Reward"][i].get<float>();
          printf("[Korali] AGENT %d/%d\n", i, nAgents);
          printf("[Korali] State: [ %.3f", state[0]);
          for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
          printf("]\n");
          printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
          printf("[Korali] Reward: %.3f\n", reward);
          printf("[Korali] Terminal?: %d\n", done);
          printf("[Korali] -------------------------------------------------------\n");
        }
      }
      else{
        auto state  = s["State"].get<std::vector<float>>();
        auto action = s["Action"].get<std::vector<float>>();
        auto reward = s["Reward"].get<float>();
        printf("[Korali] State: [ %.3f", state[0]);
        for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
        printf("]\n");
        printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
        printf("[Korali] Reward: %.3f\n", reward);
        printf("[Korali] Terminal?: %d\n", done);
        printf("[Korali] -------------------------------------------------------\n");
      }
      fflush(stdout);
    }

    // Advancing to next step
    curStep++;
  }

  // Setting termination status
  if (done == true)
    s["Termination"] = "Terminal";
  else
    s["Termination"] = "Truncated";

  // Flush CUP logger
  logger.flush();

  // Closing log file
  if( rank == 0 )
    fclose(logFile);

  // delete simulation class
  delete _environment;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);
}

bool isTerminal(StefanFish *agent, int nAgents)
{
  double xMin, xMax, yMin, yMax;
  if( nAgents == 1 ){
    // xMin = 0.6;
    // xMax = 1.4;
    xMin = 0.1;
    xMax = 1.9;

    #ifdef SWARM // uses 4x2 domain, stricter contraints
    yMin = 0.8;
    yMax = 1.2;
    #else
    // yMin = 0.2;
    // yMax = 0.8;
    yMin = 0.1;
    yMax = 0.9;
    #endif
  }
  else if( nAgents == 3 ){
    xMin = 0.4;
    xMax = 1.4;
    yMin = 0.7;
    yMax = 1.3;
  }
  else if( nAgents == 8 ){
    xMin = 0.4;
    xMax = 2.0;
    yMin = 0.6;
    yMax = 1.4;
  }
  else if( nAgents == 15 )
  {
    xMin = 0.4;
    xMax = 2.6;
    yMin = 0.5;
    yMax = 1.5;
  }
  else if( nAgents == 24 )
  {
    xMin = 0.4;
    xMax = 3.2;
    yMin = 0.4;
    yMax = 1.6;
  }
  else{
    fprintf(stderr, "Number of Agents unknown, can not finish isTerminal...\n");
    exit(-1);
  }

  const double X = agent->center[0];
  const double Y = agent->center[1];

  bool terminal = false;
  if (X < xMin) terminal = true;
  if (X > xMax) terminal = true;
  if (Y < yMin) terminal = true;
  if (Y > yMax) terminal = true;

  return terminal;
}
