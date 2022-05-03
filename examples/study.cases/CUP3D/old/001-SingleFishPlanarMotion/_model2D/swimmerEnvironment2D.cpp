//  Korali environment for CubismUP-2D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment2D.hpp"
#include "configs.hpp"

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

void runEnvironment(korali::Sample &s)
{
  // Get MPI subcommunicator, rank and size
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
  int rank, size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);
  int rankGlobal;
  MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);

  // Setting seed
  const size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory and redirecting all output to log file
  char resDir[64];
  FILE * logFile = nullptr;
  sprintf(resDir, "%s/sample%03u", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rankGlobal/size);
  if( rank == 0 )
  {
    if( not std::filesystem::exists(resDir) )
    if( not std::filesystem::create_directories(resDir) )
    {
      fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
      exit(-1);
    };
    char logFilePath[128];
    sprintf(logFilePath, "%s/log.txt", resDir);
    logFile = freopen(logFilePath, "a", stdout);
    if (logFile == NULL)
    {
      printf("[Korali] Error creating log file: %s.\n", logFilePath);
      exit(-1);
    }
  }

  // Switching to results directory
  MPI_Barrier(comm); // Make sure logfile is created before switching path
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Get task and number of agents from command line argument
  const int task = atoi(_argv[_argc-5]);
  int nAgents    = 0;

  if (task == 0)
  {
    nAgents = 1;
  }
  else
  {
      std::cerr << "Task given: " << task << " is not supported." << std::endl;
      MPI_Abort(comm,1);
  }

  Simulation *_environment = initializeEnvironment(s,task);
  // Obtain agents
  std::vector<std::shared_ptr<Shape>> shapes = _environment->getShapes();
  std::vector<StefanFish *> agents(nAgents);
  for( int i = 0; i<nAgents; i++ )
      agents[i] = dynamic_cast<StefanFish *>(shapes[i].get());

  // Setting initial state [Careful, state function needs to be called by all ranks!]
  {
    std::vector<std::vector<double>> states(nAgents);
    for(int i = 0; i<nAgents; i++ ) states[i]  = getState(agents[i]);
    if (nAgents > 1) s["State"] = states;
    else             s["State"] = states[0];
  }

  // Variables for time and step conditions
  double t        = 0; // Current time
  size_t curStep  = 0; // Current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     
  const size_t maxSteps = 50; // Max steps before truncation


  std::vector<std::vector<double>> actions(nAgents, std::vector<double>(NACTIONS));

  // Main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    if( rank == 0 ) // Get new actions, then broadcast and apply them
    {
      s.update();
      for(int i = 0; i<nAgents; i++ )
        actions[i] = (nAgents > 1) ? s["Action"][i].get<std::vector<double>>() : s["Action"].get<std::vector<double>>();
    }
    for(int i = 0; i<nAgents; i++ )
    {
      MPI_Bcast( actions[i].data(), NACTIONS, MPI_DOUBLE, 0, comm );
      agents[i]->act(t, actions[i]);
    }

    if (rank == 0) //Write a file with the actions for every agent
    {
      for( int i = 0; i<nAgents; i++ )
      {
        ofstream myfile;
        myfile.open ("actions"+std::to_string(i)+".txt",ios::app);
        myfile << t << " ";
	      for (int j = 0; j < NACTIONS ; j++)
		      myfile << actions[i][j] << " ";
	      myfile << std::endl;
        myfile.close();
      }
    }
    // Run the simulation until next action is required
    dtAct = 0.;
    for(int i = 0; i<nAgents; i++ )
    if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
      dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;
      _environment->advance(dt);

      done = _environment->sim.bCollision; // if collision -> terminate

      // Check termination because leaving margins
      for(int i = 0; i<nAgents; i++ )
        done = ( done || isTerminal(agents[i]) );
    }

    // Get and store state and reward 
    // [Careful, state function needs to be called by all ranks!] 
    {
      std::vector<std::vector<double>> states(nAgents);
      std::vector<double> rewards(nAgents);
      for(int i = 0; i<nAgents; i++ )
      {
        states[i]  = getState(agents[i]);
        rewards[i] = getReward(agents[i]);
      }
      if (nAgents > 1) { s["State"] = states   ; s["Reward"] = rewards   ;}
      else             { s["State"] = states[0]; s["Reward"] = rewards[0];}
    }

    // Printing Information:
    if ( rank == 0 )
    {
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      for(int i = 0; i<nAgents; i++ )
      {
          auto state  = (nAgents > 1) ? s["State"][i].get<std::vector<float>>():s["State"].get<std::vector<float>>();
          auto action = (nAgents > 1) ? s["Action"][i].get<std::vector<float>>():s["Action"].get<std::vector<float>>();
          auto reward = (nAgents > 1) ? s["Reward"][i].get<float>():s["Reward"].get<float>();
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
    fflush(stdout);
    curStep++;// Advance to next step
  }

  // Setting termination status
  s["Termination"] = done ? "Terminal" : "Truncated";

  logger.flush();// Flush CUP logger

  delete _environment;// delete simulation class

  if( rank == 0 ) // Closing log file
    fclose(logFile);

  std::filesystem::current_path(curPath); // Switching back to experiment directory
}

bool isTerminal(StefanFish *agent)
{
  const double xMin = 0.1;
  const double xMax = 1.9;
  const double yMin = 0.1;
  const double yMax = 1.9;
  const double X = agent->center[0];
  const double Y = agent->center[1];
  const double Xt = 0.5;
  const double Yt = 0.5;
  const double d = sqrt( (X-Xt)*(X-Xt)+(Y-Yt)*(Y-Yt) );
  if (d < 0.01) return true;
  bool terminal = false;
  if (X < xMin) terminal = true;
  if (X > xMax) terminal = true;
  if (Y < yMin) terminal = true;
  if (Y > yMax) terminal = true;
  return terminal;
}

double getReward(StefanFish *agent)
{
  const double X = agent->center[0];
  const double Y = agent->center[1];
  const double Xt = 0.5;
  const double Yt = 0.5;
  const double d = sqrt( (X-Xt)*(X-Xt)+(Y-Yt)*(Y-Yt) );
  if (d < 0.01) return 20.0;
  if (d < 0.2 ) return 0.2 - d;
  return -d;
}
std::vector<double> getState(StefanFish *agent)
{
  std::vector<double> S(7);
  S[0] = agent->center[0];
  S[1] = agent->center[1];
  S[2] = 0.0; //Z = 0
  S[3] = 0.0; //axis x-component = 0
  S[4] = 0.0; //axis y-component = 0
  S[5] = 1.0; //axis z-component = 1
  S[6] = agent->getOrientation();
  return S;
}

Simulation * initializeEnvironment(korali::Sample &s, const int task)
{
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
  int rank;
  MPI_Comm_rank(comm,&rank);
  int nAgents = 0;
  if (task == 0)
  {
    nAgents = 1;
  }
  else
  {
      std::cerr << "Task given: " << task << " is not supported." << std::endl;
      MPI_Abort(comm,1);
  }

  // Argument string to inititialize Simulation
  std::string argumentString = "CUP-RL " + OPTIONS + " -shapes ";


  /* Add Agent(s) */
  std::string AGENT = " \n\
  stefanfish L=0.2 T=1";

  // Set initial position for all agents
  for( int a = 0; a < nAgents; a++ )
  {
    std::vector<double> initialPosition = initialPositions[a];

    double initialData[3];
    initialData[0] = 0.0; //angle set to zero
    initialData[1] = initialPosition[0];
    initialData[2] = initialPosition[1];

    if ( s["Mode"] == "Training" ) // During training, add noise to inital configuration of agent
    {
      if (rank == 0) // only rank 0 samples initial data and broadcasts it
      {
        //std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
        std::uniform_real_distribution<double> disX(-0.1, 0.1);
        std::uniform_real_distribution<double> disY(-0.1, 0.1);
        //initialData[0] = initialData[0] + disA(_randomGenerator);
        initialData[1] = initialData[1] + disX(_randomGenerator);
        initialData[2] = initialData[2] + disY(_randomGenerator);
      }
      MPI_Bcast(initialData, 3, MPI_DOUBLE, 0, comm);
    }

    // Append agent to argument string
    argumentString = argumentString + AGENT + " angle=" + std::to_string(initialData[0]) + " xpos=" + std::to_string(initialData[1]) + " ypos=" + std::to_string(initialData[2]);
  }

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

  // Creating and initializing simulation environment
  Simulation *_environment = new Simulation(argv.size() - 1, argv.data(), comm);
  _environment->init();

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();
  return _environment;
}
