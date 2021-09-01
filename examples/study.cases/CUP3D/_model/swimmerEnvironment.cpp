//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment.hpp"

#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

// Swimmer following an obstacle
void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  if( not std::filesystem::exists(resDir) )
  if( not std::filesystem::create_directories(resDir) )
  {
    fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
    exit(-1);
  };

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "w", stdout);
  if (logFile == NULL)
  {
    printf("[Korali] Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Get MPI communicator
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();

  int rank;
  MPI_Comm_rank(comm,&rank);

  if( rank==0 ) {
    std::cout << "=======================================================================\n";
    std::cout << "Cubism UP 3D (velocity-pressure 3D incompressible Navier-Stokes solver)\n";
    std::cout << "=======================================================================\n";
    #ifdef NDEBUG
    std::cout << "Running in RELEASE mode!\n";
    #else
    std::cout << "Running in DEBUG mode!\n";
    #endif
  }

  // Creating simulation environment
  ArgumentParser parser(_argc, _argv);
  Simulation *_environment = new Simulation(comm, parser);

  // Obtaining agents
  std::vector<std::shared_ptr<Obstacle>> shapes = _environment->getObstacleVector();
  size_t nAgents = shapes.size() - 1;
  std::vector<StefanFish *> agents(nAgents);
  for( size_t i = 1; i<nAgents+1; i++ )
    agents[i-1] = dynamic_cast<StefanFish *>(shapes[i].get());

  // Establishing environment's dump frequency
  _environment->sim.saveTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting initial conditions
  for( size_t i = 0; i<nAgents; i++ )
    setInitialConditions(agents[i], i, s["Mode"] == "Training");
  // After moving the agent, the obstacles have to be restarted
  _environment->_init( false, parser );

  if( rank == 0 ) {
    // Setting initial state
    if( nAgents > 1 )
    {
      std::vector<std::vector<double>> states(nAgents);
      for( size_t i = 0; i<nAgents; i++ )
      {
        std::vector<double> state = agents[i]->state();
        // assign state/reward to container
        states[i]  = state;
      }
      s["State"] = states;
    }
    else
      s["State"] = agents[0]->state();
  }

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;


  // File to write actions
  std::stringstream filename;
  filename<<"actions.txt";
  ofstream myfile(filename.str().c_str());

  // Starting main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    if( rank == 0 ) {
      // Getting new actions
      s.update();

      // Reading new action(s)
      auto actions = s["Action"];

      // Setting action for each agent
      for( size_t i = 0; i<nAgents; i++ )
      {
        std::vector<double> action;
        if( nAgents > 1 )
          action = actions[i].get<std::vector<double>>();
        else
          action = actions.get<std::vector<double>>();

        // Write action to file
        if (myfile.is_open())
        {
          myfile << i << " " << action[0] << " " << action[1] << std::endl;
        }
        else{
          fprintf(stderr, "Unable to open %s file...\n", filename.str().c_str());
          exit(-1);
        }

        // Apply action
        agents[i]->act(t, action);
      }
    }

    // Run the simulation until next action is required
    if( rank == 0 ) {
      dtAct = 0.;
      for( size_t i = 0; i<nAgents; i++ )
      if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
        dtAct = agents[i]->getLearnTPeriod() * 0.5;
      tNextAct += dtAct;       
    }
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->timestep(dt);

      // Check if there was a collision -> termination.
      // TODO
      // done = _environment->sim.bCollision;

      // Check termination because leaving margins
      if ( rank == 0 ) {
        for( size_t i = 0; i<nAgents; i++ )
          done = ( done || isTerminal(agents[i], nAgents) );
      }
    }

    // Get and store state and action
    if ( rank == 0 ) {
      if( nAgents > 1 )
      {
        std::vector<std::vector<double>> states(nAgents);
        std::vector<double> rewards(nAgents);
        for( size_t i = 0; i<nAgents; i++ )
        {
          std::vector<double> state = agents[i]->state();
          // assign state/reward to container
          states[i]  = state;
          rewards[i] = done ? -10.0 : agents[i]->EffPDefBnd;
        }
        s["State"]  = states;
        s["Reward"] = rewards;
      }
      else{
        s["State"]  = agents[0]->state();
        s["Reward"] = done ? -10.0 : agents[0]->EffPDefBnd;
      }

      // Printing Information:
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      if( nAgents > 1 )
      {
        for( size_t i = 0; i<nAgents; i++ )
        {
          auto state  = s["State"][i].get<std::vector<float>>();
          auto action = s["Action"][i].get<std::vector<float>>();
          auto reward = s["Reward"][i].get<float>();
          printf("[Korali] AGENT %ld/%ld\n", i, nAgents);
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

  if ( rank == 0 ) {
    // Close file to write actions
    myfile.close();
  }

  // Flush CUP logger
  logger.flush();

  // delete simulation class
  delete _environment;

  if ( rank == 0 ) {
    // Setting finalization status
    if (done == true)
      s["Termination"] = "Terminal";
    else
      s["Termination"] = "Truncated";
  }

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining)
{
  // Initial fixed conditions
  double initialAngle = 0.0;
  std::array<double,3> initialPosition = agent->getInitialLocation();

  // with noise
  if (isTraining)
  {
    std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
    std::uniform_real_distribution<double> disX(-0.025, 0.025);
    std::uniform_real_distribution<double> disY(-0.05, 0.05);

    initialAngle = disA(_randomGenerator);
    initialPosition[0] = initialPosition[0] + disX(_randomGenerator);
    initialPosition[1] = initialPosition[1] + disY(_randomGenerator);
  }

  printf("[Korali] Initial Condition Agent %ld:\n", agentId);
  printf("[Korali] angle: %f\n", initialAngle);
  printf("[Korali] x: %f\n", initialPosition[0]);
  printf("[Korali] y: %f\n", initialPosition[1]);

  // Write initial condition to file
  std::stringstream filename;
  filename<<"initialCondition.txt";
  ofstream myfile(filename.str().c_str(), std::ofstream::app);
  if (myfile.is_open())
  {
    myfile << agentId << " " << initialAngle << " " << initialPosition[0] << " " << initialPosition[1] << std::endl;
    myfile.close();
  }
  else{
    fprintf(stderr, "Unable to open %s file...\n", filename.str().c_str());
    exit(-1);
  }

  // Setting initial position and orientation for the fish
  agent->setCenterOfMass(initialPosition);
  agent->setOrientation(initialAngle);
}

bool isTerminal(StefanFish *agent, size_t nAgents)
{
  double xMin, xMax, yMin, yMax;
  if( nAgents == 1 ){
    xMin = 0.8;
    xMax = 1.4;
    yMin = 0.8;
    yMax = 1.2;
  }
  else if( nAgents == 4 ){
    xMin = 0.4;
    xMax = 1.4;
    yMin = 0.7;
    yMax = 1.3;
  }
  else if( nAgents == 9 ){
    xMin = 0.4;
    xMax = 2.0;
    yMin = 0.6;
    yMax = 1.4;
  }
  else if( nAgents == 16 )
  {
    xMin = 0.4;
    xMax = 2.6;
    yMin = 0.5;
    yMax = 1.5;
  }
  else if( nAgents == 25 )
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

  const double X = agent->position[0];
  const double Y = agent->position[1];

  bool terminal = false;
  if (X < xMin) terminal = true;
  if (X > xMax) terminal = true;
  if (Y < yMin) terminal = true;
  if (Y > yMax) terminal = true;

  return terminal;
}
