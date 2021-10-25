#include "windmillEnvironment.hpp"
#include "Operators/Helpers.h"
#include "Definitions.h"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

// 4 windmills with variable torque applied to them
void runEnvironment(korali::Sample &s)
{
  std::cout<<"Before state env"<<std::endl;
  ////////////////////////////////////////// setup stuff 
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  std::filesystem::create_directories(resDir);

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv);
  _environment->init();
  ////////////////////////////////////////// setup stuff 

  ////////////////////////////////////////// Initialize agents and objective
  // Obtaining agent, 4 windmills 
  Windmill* agent1 = dynamic_cast<Windmill*>(_environment->getShapes()[0]);
  Windmill* agent2 = dynamic_cast<Windmill*>(_environment->getShapes()[1]);

  // useful agent functions :
  // void act( double action );
  // double reward( std::array<Real,2> target, std::vector<double> target_vel, double C = 10);
  // std::vector<double> state();

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting initial conditions
  setInitialConditions(agent1, 0.0, s["Mode"] == "Training");
  setInitialConditions(agent2, 0.0, s["Mode"] == "Training");
  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Set target
  std::array<Real,2> target_pos{0.7,0.6};
  std::array<Real, 2> target_vel={0.0,0.0};

  agent1->setTarget(target_pos);
  agent2->setTarget(target_pos);

  std::vector<double> state1 = agent1->state();
  std::vector<double> state2 = agent2->state();

  //std::vector<double> state = {state1[0], state1[1], state2[0], state2[1]};

  std::vector<double> center_area = {target_pos[0], target_pos[1]};
  std::vector<double> dim = {0.2, 0.2};

  std::cout<<"Before state "<<std::endl;

  //std::vector<double> state = getConvState(_environment, center_area);
  std::vector<double> state = getUniformGridVort(_environment);

  s["State"] = state;

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 2000; // 2000 for training

  while (curStep < maxSteps)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action for 
    agent1->act( action[0] );
    agent2->act( action[1] );

    // Run the simulation until next action is required
    tNextAct += 0.01;
    while ( t < tNextAct )
    {
      // Calculate simulation timestep
      const double dt = std::min(_environment->calcMaxTimestep(), 0.01);
      t += dt;

      // Advance simulation
      _environment->advance(dt);
    }

    // used for "both" run
    // Real en = 5.0e4;
    // Real flow = 2.5;

    // used for "energy_zero" run
    // Real en = 5.0e4;
    // Real flow = 0.0;

    // used for "flow_zero" run
    Real en = 0.0;
    Real flow = 2.5;

    double r1 = agent1->reward( target_vel,  en, flow);
    double r2 = agent2->reward( target_vel,  en, flow);
    double reward = (r1 + r2);

    // // Obtaining new agent state
    // state1 = agent1->state();
    // state2 = agent2->state();
    // state = {state1[0], state1[1], state2[0], state2[1]};

    // // Printing Information:
    // printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    // printf("[Korali] State: [ ");
    // for (size_t i = 0; i < state.size(); i++){
    //   if (i%2 == 0){
    //     printf("[%.3f, ", state[i]);
    //   } else {
    //     printf("%.3f]", state[i]);
    //   }
    // }
    // printf("]\n");
    // printf("[Korali] Action: [ %.8f, %.8f ]\n", action[0], action[1]);
    // printf("[Korali] Reward: %.3f+%.3f=%.3f\n", r1, r2, reward);
    // printf("[Korali] -------------------------------------------------------\n");
    // fflush(stdout);

    // Storing reward
    s["Reward"] = reward;

    state = getConvState(_environment, center_area);

    // Storing new state
    s["State"] = state;

    // Advancing to next step
    curStep++;
  }

  // Flush CUP logger
  logger.flush();

  // delete simulation class
  delete _environment;

  // Setting finalization status
  s["Termination"] = "Truncated";

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
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

// get the grid state for case with convnet as input

std::vector<double> getConvState(Simulation *_environment, std::vector<double> pos)
{
  std::vector<double> vorticities;
  // get the simulation data
  const auto K1 = computeVorticity(_environment->sim); K1.run();

  // the vorticity is stored in the ScalarGrid* tmp
  // get the part of the grid that interests us
  // we will get all the blocks that together contain the area fully
  const std::vector<cubism::BlockInfo>& vortInfo = _environment->sim.tmp->getBlocksInfo();

  int bpdx = _environment->sim.bpdx;
  int bpdy = _environment->sim.bpdy;

  size_t num_points_x = ScalarBlock::sizeX;
  size_t num_points_y = ScalarBlock::sizeY;

  int index = 0;

  // loop over all the blocks
  for(size_t t=0; t < vortInfo.size(); ++t)
  {
    const cubism::BlockInfo & info = vortInfo[t];
    
    // find block associated with the center_area point
    // get gridspacing in block
    const Real h = info.h_gridpoint;

    // compute lower left corner of block
    std::array<Real,2> MIN = info.pos<Real>(0, 0);
    for(int j=0; j<2; ++j)
      MIN[j] -= 0.5 * h; // pos returns cell centers

    // compute top right corner of block
    std::array<Real,2> MAX = info.pos<Real>(num_points_x-1, num_points_y-1);
    for(int j=0; j<2; ++j)
      MAX[j] += 0.5 * h; // pos returns cell centers

    // check whether point is inside block
    if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
    {
      // select block
      index = t;
    }

  }

  // get all the blocks surrounding the block that contains the point
  const cubism::BlockInfo & info = vortInfo[index];

  const cubism::BlockInfo & west  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(-1, 0, 0));
  const cubism::BlockInfo & east  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(1, 0, 0));
  const cubism::BlockInfo & north  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(0, 1, 0));
  const cubism::BlockInfo & south  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(0, -1, 0));

  const cubism::BlockInfo & ne  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(1, 1, 0));
  const cubism::BlockInfo & nw  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(-1, 1, 0));
  const cubism::BlockInfo & sw  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(-1, -1, 0));
  const cubism::BlockInfo & se  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(1, -1, 0));

  const std::vector<std::vector<cubism::BlockInfo>>& blocks = {{nw, north, ne}, {west, info, east}, {sw, south, se}};

  for(size_t b_i = 0; b_i < 3; ++b_i)
  for(size_t b_j = 0; b_j < 3; ++b_j)
  for(size_t i = 0; i < num_points_x; ++i)
  for(size_t j = 0; j < num_points_y; ++j)
  {
    const ScalarBlock& b = * (const ScalarBlock*) blocks[b_i][b_j].ptrBlock;
    
    vorticities.push_back(b(i,j).s);
  }

  /////////////////////////////////////////////////////////


  // size_t num_points = ScalarBlock::sizeY;

  // // loop over the rows
  // for(size_t j=0; j < num_points; ++j)
  // {
  //   // loop over all the blocks
  //   for(size_t t=0; t < vortInfo.size(); ++t)
  //   {
  //     // get pointer on block
  //     const ScalarBlock& b = * (const ScalarBlock*) vortInfo[t].ptrBlock;

  //     // loop over the cols
  //     for(size_t i=0; i < b.sizeX; ++i)
  //       {
  //         const std::array<Real,2> oSens = vortInfo[t].pos<Real>(i, j);
  //         if (isInConvArea(oSens, center_area, dim))
  //         {
  //           vorticities.push_back(b(i,j).s);
  //         }
  //       }
  //   }
  // }

  std::cout<<"size of vorticities: "<<vorticities.size()<<std::endl;

  return vorticities;
}

bool isInConvArea(const std::array<Real,2> point, std::vector<double> target, std::vector<double> dimensions)
{
  std::array<Real, 2> lower_left = {target[0]-dimensions[0]/2.0, target[1]-dimensions[1]/2.0};
  std::array<Real, 2> upper_right = {target[0]+dimensions[0]/2.0, target[1]+dimensions[1]/2.0};

  if(point[0] >= lower_left[0] && point[0] <= upper_right[0])
  {
    if(point[1] >= lower_left[1] && point[1] <= upper_right[1])
    {
      return true;
    }
  }

  return false;
}



std::vector<double> getUniformGridVort(Simulation *_environment)
{
  const unsigned int nX = ScalarBlock::sizeX;
  const unsigned int nY = ScalarBlock::sizeY;

  const auto K1 = computeVorticity(_environment->sim); K1.run();

  const std::vector<cubism::BlockInfo>& vortInfo = _environment->sim.tmp->getBlocksInfo();

  const int levelMax = _environment->sim.tmp->getlevelMax();

  std::array<int, 3> bpd = _environment->sim.tmp->getMaxBlocks();
  const unsigned int unx = bpd[0]*(1<<(levelMax-1))*nX; // maximum number of blocks in x directions
  const unsigned int uny = bpd[1]*(1<<(levelMax-1))*nY; // same in y direction


  std::vector<double> uniform_mesh(uny*unx); // vector containing all the points in the simulation

  // loop over the blocks
  for (size_t i = 0 ; i < vortInfo.size() ; i++)
  {
    const int level = vortInfo[i].level;
    const cubism::BlockInfo & info = vortInfo[i];
    const ScalarBlock& b = * (const ScalarBlock*) info.ptrBlock;

    for (unsigned int y = 0; y < nY; y++)
    for (unsigned int x = 0; x < nX; x++)
    {
      double output = 0.0;
      double dudx = 0.0;
      double dudy = 0.0;

      output = b(x,y).s;
      if (x!= 0 && x!= nX-1)
      {
        double output_p = 0.0;
        double output_m = 0.0;
        output_p = b(x+1, y).s;
        output_m = b(x-1, y).s;
        dudx = 0.5*(output_p-output_m);
      }
      else if (x==0)
      {
        double output_p = 0.0;
        output_p = b(x+1,y).s;
        dudx = output_p-output;   
      }
      else
      {
        double output_m = 0.0;
        output_m = b(x-1, y).s;
        dudx = output-output_m;     
      }

      if (y!= 0 && y!= nY-1)
      {
        double output_p = 0.0;
        double output_m = 0.0;
        output_p = b(x, y+1).s;
        output_m = b(x, y-1).s;
        dudy = 0.5*(output_p-output_m);
      }
      else if (y==0)
      {
        double output_p = 0.0;
        output_p = b(x, y+1).s;
        dudy = output_p-output;     
      }
      else
      {
        double output_m = 0.0;
        output_m = b(x, y-1).s;
        dudy = output-output_m;       
      }

      // refinement part

      int iy_start = (info.index[1]*nY + y)*(1<< ( (levelMax-1)-level ) );
      int ix_start = (info.index[0]*nX + x)*(1<< ( (levelMax-1)-level ) );

      const int points = 1<< ( (levelMax-1)-level ); 
      const double dh = 1.0/points;

      for (int iy = iy_start; iy< iy_start + (1<< ( (levelMax-1)-level ) ); iy++)
      for (int ix = ix_start; ix< ix_start + (1<< ( (levelMax-1)-level ) ); ix++)
      {
        double cx = (ix - ix_start - points/2 + 1 - 0.5)*dh;
        double cy = (iy - iy_start - points/2 + 1 - 0.5)*dh;
        uniform_mesh[iy*unx+ix] = output + cx*dudx + cy*dudy;
        // for (unsigned int j = 0; j < NCHANNELS; ++j)
        //   uniform_mesh[iy*NCHANNELS*unx+ix*NCHANNELS+j] = output[j]+ cx*dudx[j]+ cy*dudy[j];
      }
    }
  }

  std::cout<<"Total size of grid: "<<uniform_mesh.size()<<std::endl;

  return uniform_mesh;

}