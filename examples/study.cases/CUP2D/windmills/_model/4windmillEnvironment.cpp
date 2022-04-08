#include "windmillEnvironment.hpp"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;

Simulation *_environment;
std::mt19937 _randomGenerator;

// 4 windmills with variable torque applied to them
void runEnvironment(korali::Sample &s)
{
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
  Windmill* agent3 = dynamic_cast<Windmill*>(_environment->getShapes()[2]);
  Windmill* agent4 = dynamic_cast<Windmill*>(_environment->getShapes()[3]);

  // useful agent functions :
  // void act( double action );
  // double reward( std::array<Real,2> target, std::vector<double> target_vel, double C = 10);
  // std::vector<double> state();

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Reseting environment and setting initial conditions
  _environment->reset();
  bool random_init = (s["Mode"] == "Training");
  setInitialConditions(agent1, 0, random_init);
  setInitialConditions(agent2, 0, random_init);
  setInitialConditions(agent3, 0, random_init);
  setInitialConditions(agent4, 0, random_init);
  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Set target
  std::array<Real,2> target_pos{0.9,0.7};
  std::array<double, 2> target_vel={0.0,0.0};

  agent1->setTarget(target_pos);
  agent2->setTarget(target_pos);
  agent3->setTarget(target_pos);
  agent4->setTarget(target_pos);

  std::vector<double> state1 = agent1->state();
  std::vector<double> state2 = agent2->state();
  std::vector<double> state3 = agent3->state();
  std::vector<double> state4 = agent4->state();

  std::vector<double> state = {state1[0], state1[1], state2[0], state2[1], state3[0], state3[1], state4[0], state4[1]};

  s["State"] = state;

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 2000;

  // Starting main environment loop
  // bool done = false;

  while (curStep < maxSteps)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action
    agent1->act( action[0] );
    agent2->act( action[1] );
    agent3->act( action[2] );
    agent4->act( action[3] );

    // Run the simulation until next action is required
    tNextAct += 0.01;
    while ( t < tNextAct )
    {
      // Advance simulation
      const double dt = _environment->calcMaxTimestep();
      t += dt;

      // Advance simulation and check whether it is correct
      _environment->advance(dt);
    }

    // reward( std::vector<double> target, std::vector<double> target_vel, double C = 10)
    Real en = 0.0;
    Real flow = 2.5;

    double r1 = agent1->reward( target_vel,  en, flow);
    double r2 = agent2->reward( target_vel,  en, flow);
    double r3 = agent3->reward( target_vel,  en, flow);
    double r4 = agent4->reward( target_vel,  en, flow);
    double reward = (r1 + r2 + r3 + r4);

    // Obtaining new agent state
    state1 = agent1->state();
    state2 = agent2->state();
    state3 = agent3->state();
    state4 = agent4->state();
    state = {state1[0], state1[1], state2[0], state2[1], state3[0], state3[1], state4[0], state4[1]};

    // Printing Information:
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ ");
    for (size_t i = 0; i < state.size(); i++){
      if (i%2 == 0){
        printf("[%.3f, ", state[i]);
      } else {
        printf("%.3f]", state[i]);
      }
    }
    printf("]\n");
    printf("[Korali] Action: [ %.8f, %.8f, %.8f, %.8f  ]\n", action[0], action[1], action[2], action[3]);
    printf("[Korali] Reward: %.3f\n", reward);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);

    // Storing reward
    s["Reward"] = reward;

    // Storing new state
    s["State"] = state;

    // Advancing to next step
    curStep++;
  }

  // Setting finalization status
  // if (done == true)
  //   s["Termination"] = "Terminal";
  // else
  //   s["Termination"] = "Truncated";
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



// old stuff

// from fully refined grid (simulation at level 1 fo refinement), get the block and the 8 neighbouring blocks next to
// the pos. Essentially very slow as the simulation has to be full refined for this to work

// std::vector<double> getConvState(Simulation *_environment, std::vector<double> pos)
// {
//   std::vector<double> vorticities;
//   // get the simulation data
//   const auto K1 = computeVorticity(_environment->sim); K1.run();

//   // the vorticity is stored in the ScalarGrid* tmp
//   // get the part of the grid that interests us
//   // we will get all the blocks that together contain the area fully
//   const std::vector<cubism::BlockInfo>& vortInfo = _environment->sim.tmp->getBlocksInfo();

//   int bpdx = _environment->sim.bpdx;
//   int bpdy = _environment->sim.bpdy;

//   size_t num_points_x = ScalarBlock::sizeX;
//   size_t num_points_y = ScalarBlock::sizeY;

//   int index = 0;

//   // get index of block containing point "pos"
//   // loop over all the blocks
//   for(size_t t=0; t < vortInfo.size(); ++t)
//   {
//     const cubism::BlockInfo & info = vortInfo[t];
    
//     // find block associated with the center_area point
//     // get gridspacing in block
//     const Real h = info.h_gridpoint;

//     // compute lower left corner of block
//     std::array<Real,2> MIN = info.pos<Real>(0, 0);
//     for(int j=0; j<2; ++j)
//       MIN[j] -= 0.5 * h; // pos returns cell centers

//     // compute top right corner of block
//     std::array<Real,2> MAX = info.pos<Real>(num_points_x-1, num_points_y-1);
//     for(int j=0; j<2; ++j)
//       MAX[j] += 0.5 * h; // pos returns cell centers

//     // check whether point is inside block
//     if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
//     {
//       // select block
//       index = t;
//     }

//   }

//   // get all the blocks surrounding the block that contains the point
//   const cubism::BlockInfo & info = vortInfo[index];

//   const cubism::BlockInfo & west  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(-1, 0, 0));
//   const cubism::BlockInfo & east  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(1, 0, 0));
//   const cubism::BlockInfo & north  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(0, 1, 0));
//   const cubism::BlockInfo & south  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(0, -1, 0));

//   const cubism::BlockInfo & ne  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(1, 1, 0));
//   const cubism::BlockInfo & nw  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(-1, 1, 0));
//   const cubism::BlockInfo & sw  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(-1, -1, 0));
//   const cubism::BlockInfo & se  = _environment->sim.tmp->getBlockInfoAll(info.level, info.Znei_(1, -1, 0));

//   const std::vector<std::vector<cubism::BlockInfo>>& blocks = {{nw, north, ne}, {west, info, east}, {sw, south, se}};

//   for(size_t b_i = 0; b_i < 3; ++b_i)
//   for(size_t b_j = 0; b_j < 3; ++b_j)
//   for(size_t i = 0; i < num_points_x; ++i)
//   for(size_t j = 0; j < num_points_y; ++j)
//   {
//     const ScalarBlock& b = * (const ScalarBlock*) blocks[b_i][b_j].ptrBlock;
    
//     vorticities.push_back(b(i,j).s);
//   }

//   /////////////////////////////////////////////////////////


//   // size_t num_points = ScalarBlock::sizeY;

//   // // loop over the rows
//   // for(size_t j=0; j < num_points; ++j)
//   // {
//   //   // loop over all the blocks
//   //   for(size_t t=0; t < vortInfo.size(); ++t)
//   //   {
//   //     // get pointer on block
//   //     const ScalarBlock& b = * (const ScalarBlock*) vortInfo[t].ptrBlock;

//   //     // loop over the cols
//   //     for(size_t i=0; i < b.sizeX; ++i)
//   //       {
//   //         const std::array<Real,2> oSens = vortInfo[t].pos<Real>(i, j);
//   //         if (isInConvArea(oSens, center_area, dim))
//   //         {
//   //           vorticities.push_back(b(i,j).s);
//   //         }
//   //       }
//   //   }
//   // }

//   std::cout<<"size of vorticities: "<<vorticities.size()<<std::endl;

//   return vorticities;
// }

// bool isInConvArea(const std::array<Real,2> point, std::vector<double> target, std::vector<double> dimensions)
// {
//   std::array<Real, 2> lower_left = {target[0]-dimensions[0]/2.0, target[1]-dimensions[1]/2.0};
//   std::array<Real, 2> upper_right = {target[0]+dimensions[0]/2.0, target[1]+dimensions[1]/2.0};

//   if(point[0] >= lower_left[0] && point[0] <= upper_right[0])
//   {
//     if(point[1] >= lower_left[1] && point[1] <= upper_right[1])
//     {
//       return true;
//     }
//   }

//   return false;
// }


// // From the simulation, get the uniform grid of vorticities.

// std::vector<double> getUniformGridVort(Simulation *_environment, const std::array<Real,2> pos)
// {
//   const unsigned int nX = ScalarBlock::sizeX;
//   const unsigned int nY = ScalarBlock::sizeY;

//   const auto K1 = computeVorticity(_environment->sim); K1.run();

//   const std::vector<cubism::BlockInfo>& vortInfo = _environment->sim.tmp->getBlocksInfo();

//   const int levelMax = _environment->sim.tmp->getlevelMax();

//   std::array<int, 3> bpd = _environment->sim.tmp->getMaxBlocks();
//   const unsigned int unx = bpd[0]*(1<<(levelMax-1))*nX; // maximum number of points in x directions, fully refined
//   const unsigned int uny = bpd[1]*(1<<(levelMax-1))*nY; // same in y direction

//   // need to change this if not square grid
//   // double extent = _environment_>sim.extent; // extent of dim with most bpd
//   // double h_min = extent / std::max(unx, uny);
//   // double h_max


//   //std::vector<double> uniform_mesh(uny*unx); // vector containing all the points in the simulation
//   std::vector<std::vector<double>> uniform_mesh (uny, std::vector<double> (unx, 0));

//   // info and block containing the pos point
//   size_t id = holdingBlockID(pos, vortInfo);
//   const cubism::BlockInfo & point_info = vortInfo[id];
//   const ScalarBlock& point_b = * (const ScalarBlock*) point_info.ptrBlock;

//   // get origin of block
//   const std::array<Real,2> oSens = point_info.pos<Real>(0, 0);

//   // get inverse gridspacing in block
//   const Real invh = 1/(point_info.h_gridpoint);

//   // get index for sensor
//   const std::array<int,2> iSens = safeIdInBlock(pos, oSens, invh);

//   const int level_block = point_info.level;

//   int N_block = (1<< ( (levelMax-1)-level_block ) );
//   int i_point = (point_info.index[0]*nX + iSens[0])*N_block;
//   int j_point = (point_info.index[1]*nY + iSens[1])*N_block; // gets coordinate of point for fully refined grid
  
//   // std::cout<<"Before loop"<<std::endl;


//   // loop over the blocks
//   for (size_t i = 0 ; i < vortInfo.size() ; i++)
//   {
//     const int level = vortInfo[i].level;
//     const cubism::BlockInfo & info = vortInfo[i];
//     const ScalarBlock& b = * (const ScalarBlock*) info.ptrBlock;

//     const Real h = vortInfo[i].h_gridpoint;

//     for (unsigned int y = 0; y < nY; y++)
//     for (unsigned int x = 0; x < nX; x++)
//     {
//       double output = 0.0;
//       double dudx = 0.0;
//       double dudy = 0.0;

//       output = b(x,y).s;
//       if (x!= 0 && x!= nX-1)
//       {
//         double output_p = 0.0;
//         double output_m = 0.0;
//         output_p = b(x+1, y).s;
//         output_m = b(x-1, y).s;
//         dudx = 0.5*(output_p-output_m);
//       }
//       else if (x==0)
//       {
//         double output_p = 0.0;
//         output_p = b(x+1,y).s;
//         dudx = output_p-output;   
//       }
//       else
//       {
//         double output_m = 0.0;
//         output_m = b(x-1, y).s;
//         dudx = output-output_m;     
//       }

//       if (y!= 0 && y!= nY-1)
//       {
//         double output_p = 0.0;
//         double output_m = 0.0;
//         output_p = b(x, y+1).s;
//         output_m = b(x, y-1).s;
//         dudy = 0.5*(output_p-output_m);
//       }
//       else if (y==0)
//       {
//         double output_p = 0.0;
//         output_p = b(x, y+1).s;
//         dudy = output_p-output;     
//       }
//       else
//       {
//         double output_m = 0.0;
//         output_m = b(x, y-1).s;
//         dudy = output-output_m;       
//       }

//       // refinement part
//       // for each bloc, indepent of how refined they are, fully refine them

//       // index[] //(i,j,k) coordinates of block at given refinement level, i.e.,
//       // if the entire grid was at the same refinement level, what would be the index of the point
//       // nY and nY = # of points per direction in each block
//       int N = (1<< ( (levelMax-1)-level ) ); // number of points to add per point so that grid is fully refined
//       int iy_start = (info.index[1]*nY + y)*N; // gets coordinate of point for fully refined grid
//       int ix_start = (info.index[0]*nX + x)*N;


//       // const int points = 1<< ( (levelMax-1)-level ); // multiply 1 by 2^(levelMax-1 - level), i.e. number of points per direction
//       const double dh = 1.0/N; // grid spacing

//       for (int iy = iy_start; iy< iy_start + N; iy++)
//       for (int ix = ix_start; ix< ix_start + N; ix++)
//       {
//         double cx = (ix - ix_start - N/2 + 1 - 0.5)*dh;
//         double cy = (iy - iy_start - N/2 + 1 - 0.5)*dh;
//         //uniform_mesh[iy*unx+ix] = output + cx*dudx + cy*dudy;
//         uniform_mesh[iy][ix] = output + cx*dudx + cy*dudy;
//         // for (unsigned int j = 0; j < NCHANNELS; ++j)
//         //   uniform_mesh[iy*NCHANNELS*unx+ix*NCHANNELS+j] = output[j]+ cx*dudx[j]+ cy*dudy[j];
//       }


//     }
//   }

//   // std::cout<<"After loop"<<std::endl;

//   std::vector<double> subsample(576);
//   int ind = 0;

//   for(int i = -12; i < 12; ++i)
//   for(int j = -12; j < 12; ++j)
//   {
//     subsample[ind] = uniform_mesh[i_point + i][j_point + j];
//     ind+=1;
//   }


//   //std::cout<<"Total size of grid: "<<uniform_mesh.size()<<std::endl;

//   return subsample;

// }

// std::vector<double> getUniformLevelGridVort(Simulation *_environment, const std::array<Real,2> pos, int grid_level)
// {
//   // this function gets the uniform grid at a certain level
//   // works similarly to the getUniformGridVort except it averages the vorticities if the block is too refined
//   const unsigned int nX = ScalarBlock::sizeX;
//   const unsigned int nY = ScalarBlock::sizeY;

//   const auto K1 = computeVorticity(_environment->sim); K1.run();

//   const std::vector<cubism::BlockInfo>& vortInfo = _environment->sim.tmp->getBlocksInfo();

//   const int levelMax = _environment->sim.tmp->getlevelMax();

//   std::array<int, 3> bpd = _environment->sim.tmp->getMaxBlocks();


//   ////// 
//   const unsigned int unx = bpd[0]*(1<<(grid_level-1))*nX; // maximum number of points in x directions, refined to grid_level
//   const unsigned int uny = bpd[1]*(1<<(grid_level-1))*nY; // same in y direction

//   // need to change this if not square grid
//   // double extent = _environment_>sim.extent; // extent of dim with most bpd
//   // double h_min = extent / std::max(unx, uny);
//   // double h_max


//   //std::vector<double> uniform_mesh(uny*unx); // vector containing all the points in the simulation
//   std::vector<std::vector<double>> uniform_mesh (uny, std::vector<double> (unx, 0));

//   // loop over the blocks
//   for (size_t i = 0 ; i < vortInfo.size() ; i++)
//   {
//     const int level = vortInfo[i].level;
//     const cubism::BlockInfo & info = vortInfo[i];
//     const ScalarBlock& b = * (const ScalarBlock*) info.ptrBlock;

//     // grid is more refined that it needs to be
//     // average the values
//     if(level+1 == grid_level+1)
//     {
//       for (unsigned int y = 0; y < nY; y+=2)
//       for (unsigned int x = 0; x < nX; x+=2)
//       {
//         double avg = b(x, y).s + b(x+1, y).s + b(x, y+1).s + b(x+1,y+1).s;

//         int N = 2;
        
//         int iy = (info.index[1]*nY + y)/N; // gets coordinate of point for fully refined grid
//         int ix = (info.index[0]*nX + x)/N;

//         uniform_mesh[iy][ix] = avg;
//       }

//     } else if (level+1 == grid_level) // grid is at same level of refinement
//     {
//       for (unsigned int y = 0; y < nY; y++)
//       for (unsigned int x = 0; x < nX; x++)
//       {
//         if (level == grid_level) // get indices of points
//         {
//           int N = 2;
          
//           int iy = (info.index[1]*nY + y); // gets coordinate of point for fully refined grid
//           int ix = (info.index[0]*nX + x);

//           uniform_mesh[iy][ix] = b(x, y).s;
//         }
//       }
//     }
//     else // grid needs to be refined
//     {
//       const Real h = vortInfo[i].h_gridpoint;

//       for (unsigned int y = 0; y < nY; y++)
//       for (unsigned int x = 0; x < nX; x++)
//       {
//         double output = 0.0;
//         double dudx = 0.0;
//         double dudy = 0.0;

//         output = b(x,y).s;
//         if (x!= 0 && x!= nX-1)
//         {
//           double output_p = 0.0;
//           double output_m = 0.0;
//           output_p = b(x+1, y).s;
//           output_m = b(x-1, y).s;
//           dudx = 0.5*(output_p-output_m);
//         }
//         else if (x==0)
//         {
//           double output_p = 0.0;
//           output_p = b(x+1,y).s;
//           dudx = output_p-output;   
//         }
//         else
//         {
//           double output_m = 0.0;
//           output_m = b(x-1, y).s;
//           dudx = output-output_m;     
//         }

//         if (y!= 0 && y!= nY-1)
//         {
//           double output_p = 0.0;
//           double output_m = 0.0;
//           output_p = b(x, y+1).s;
//           output_m = b(x, y-1).s;
//           dudy = 0.5*(output_p-output_m);
//         }
//         else if (y==0)
//         {
//           double output_p = 0.0;
//           output_p = b(x, y+1).s;
//           dudy = output_p-output;     
//         }
//         else
//         {
//           double output_m = 0.0;
//           output_m = b(x, y-1).s;
//           dudy = output-output_m;       
//         }

//         // refinement part
//         // for each bloc, indepent of how refined they are, fully refine them

//         // index[] //(i,j,k) coordinates of block at given refinement level, i.e.,
//         // if the entire grid was at the same refinement level, what would be the index of the point
//         // nY and nY = # of points per direction in each block
//         int N = (1<< ( (grid_level-1)-level ) ); // number of points to add per point so that grid is fully refined
//         int iy_start = (info.index[1]*nY + y)*N; // gets coordinate of point for fully refined grid
//         int ix_start = (info.index[0]*nX + x)*N;


//         // const int points = 1<< ( (levelMax-1)-level ); // multiply 1 by 2^(levelMax-1 - level), i.e. number of points per direction
//         const double dh = 1.0/N; // grid spacing

//         for (int iy = iy_start; iy< iy_start + N; iy++)
//         for (int ix = ix_start; ix< ix_start + N; ix++)
//         {
//           double cx = (ix - ix_start - N/2 + 1 - 0.5)*dh;
//           double cy = (iy - iy_start - N/2 + 1 - 0.5)*dh;
//           //uniform_mesh[iy*unx+ix] = output + cx*dudx + cy*dudy;
//           uniform_mesh[iy][ix] = output + cx*dudx + cy*dudy;
//           // for (unsigned int j = 0; j < NCHANNELS; ++j)
//           //   uniform_mesh[iy*NCHANNELS*unx+ix*NCHANNELS+j] = output[j]+ cx*dudx[j]+ cy*dudy[j];
//         }


//       }
//     }
//   }


//   // info and block containing the pos point
//   size_t id = holdingBlockID(pos, vortInfo);
//   const cubism::BlockInfo & point_info = vortInfo[id];
//   const ScalarBlock& point_b = * (const ScalarBlock*) point_info.ptrBlock;

//   // get origin of block
//   const std::array<Real,2> oSens = point_info.pos<Real>(0, 0);

//   // get inverse gridspacing in block
//   const Real invh = 1/(point_info.h_gridpoint);

//   // get index for sensor
//   const std::array<int,2> iSens = safeIdInBlock(pos, oSens, invh);

//   const int level_block = point_info.level;

//   int N_block = (1<< ( (grid_level-1)-level_block ) );
//   int i_point = (point_info.index[0]*nX + iSens[0]);
//   int j_point = (point_info.index[1]*nY + iSens[1]);
//   if (level_block+1==grid_level+1){
//     i_point /= 2;
//     j_point /= 2;
//   } else if (level_block+1 == grid_level)
//   {
//     // pass
//   } else
//   {
//     i_point *= N_block;
//     j_point *= N_block;
//   }

//   // std::cout<<"After loop"<<std::endl;

//   std::vector<double> subsample(576);
//   int ind = 0;

//   for(int i = -12; i < 12; ++i)
//   for(int j = -12; j < 12; ++j)
//   {
//     subsample[ind] = uniform_mesh[i_point + i][j_point + j];
//     ind+=1;
//   }


//   //std::cout<<"Total size of grid: "<<uniform_mesh.size()<<std::endl;

//   return subsample;

// }

// std::vector<double> vortGridProfile(Simulation *_environment)
// {
//   std::vector<double> vort_profile(576, 0.0);

//   double height = 0.021875;
//   double region_area = height * height;

//   const std::vector<cubism::BlockInfo>& velInfo = _environment->sim.tmp->getBlocksInfo();
  
//   // loop over all the blocks
//   for(size_t t=0; t < velInfo.size(); ++t)
//   {
//     // get pointer on block
//     const ScalarBlock& b = * (const ScalarBlock*) velInfo[t].ptrBlock;
   
//     double da = velInfo[t].h_gridpoint * velInfo[t].h_gridpoint;
//     // loop over all the points
//     for(size_t i=0; i < b.sizeX; ++i)
//     {
//       for(size_t j=0; j < b.sizeY; ++j)
//       {
//         const std::array<Real,2> oSens = velInfo[t].pos<Real>(i, j);
//         std::vector<int> num = idRegion(oSens, height);
//         if (num[0] == 1)
//         {
//           int index = 24 * (num[1]-1) + num[2] - 1;
//           vort_profile[index] += b(i,j).s * da;
//         }
//       }
//     }
//   }



//   // divide each vel_avg by the corresponding area

//   for (int k = 0; k < 24; ++k)
//   {
//     for (int p = 0; p < 24; ++p)
//     {
//       vort_profile[24*k + p] /= region_area;
//     }
//   }

//   return vort_profile;
// }

// std::vector<double> velGridProfile(Simulation *_environment)
// {
//   std::vector<double> vel_profile(1024, 0.0);

//   double height = 0.021875;
//   double region_area = height * height;

//   const std::vector<cubism::BlockInfo>& velInfo = _environment->sim.vel->getBlocksInfo();
  
//   // loop over all the blocks
//   for(size_t t=0; t < velInfo.size(); ++t)
//   {
//     // get pointer on block
//     const VectorBlock& b = * (const VectorBlock*) velInfo[t].ptrBlock;
   
//     double da = velInfo[t].h_gridpoint * velInfo[t].h_gridpoint;
//     // loop over all the points
//     for(size_t i=0; i < b.sizeX; ++i)
//     {
//       for(size_t j=0; j < b.sizeY; ++j)
//       {
//         const std::array<Real,2> oSens = velInfo[t].pos<Real>(i, j);
//         std::vector<int> num = idRegion(oSens, height);
//         if (num[0] == 1)
//         {
//           int index = 24 * (num[1]-1) + num[2] - 1;
//           vel_profile[index] += b(i, j).u[0] * da;
//         }
//       }
//     }
//   }

//   // divide each vel_avg by the corresponding area

//   for (int k = 0; k < 32; ++k)
//   {
//     for (int p = 0; p < 32; ++p)
//     {
//       vel_profile[32*k + p] /= region_area;
//     }
//   }

//   return vel_profile;
// }

// std::vector<int> idRegion(const std::array<Real, 2> point, double height)
// {
//   // returns 0 if outside of the box
//   // std::array<Real, 2> lower_left = {0.525, 0.35};
//   // std::array<Real, 2> upper_right = {1.05, 0.875};

//   std::array<Real, 2> lower_left = {0.35, 0.35};
//   std::array<Real, 2> upper_right = {1.05, 1.05};


//   double rel_pos_height = point[1] - lower_left[1];
//   double rel_pos_width = point[0] - lower_left[0];


//   std::vector<int> num(3, 0);

//   if(point[0] >= lower_left[0] && point[0] <= upper_right[0])
//   {
//     if(point[1] >= lower_left[1] && point[1] <= upper_right[1])
//     {
//       // point is inside the rectangle to compute velocity profile
//       // now find out in what region of the rectangle we are in
//       num[0] = 1;
//       num[1] = static_cast<int>(std::ceil(rel_pos_height/height));
//       num[2] = static_cast<int>(std::ceil(rel_pos_width/height));
//       //std::cout<<"num "<<num[1]<<" "<<num[2]<<std::endl;
      
//       return num;
//     }
//   }

//   return num;
// }

// // function that finds block id of block containing pos (x,y)
// size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo)
// {
//   for(size_t i=0; i<velInfo.size(); ++i)
//   {
//     // get gridspacing in block
//     const Real h = velInfo[i].h_gridpoint;

//     // compute lower left corner of block
//     std::array<Real,2> MIN = velInfo[i].pos<Real>(0, 0);
//     for(int j=0; j<2; ++j)
//       MIN[j] -= 0.5 * h; // pos returns cell centers

//     // compute top right corner of block
//     std::array<Real,2> MAX = velInfo[i].pos<Real>(VectorBlock::sizeX-1, VectorBlock::sizeY-1);
//     for(int j=0; j<2; ++j)
//       MAX[j] += 0.5 * h; // pos returns cell centers

//     // check whether point is inside block
//     if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
//     {
//       // select block
//       return i;
//     }
//   }
//   printf("ABORT: coordinate (%g,%g) could not be associated to block\n", pos[0], pos[1]);
//   fflush(0); abort();
//   return 0;
// };

// // function that gives indice of point in block
// std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh )
// {
//   const int indx = (int) std::round((pos[0] - org[0])*invh);
//   const int indy = (int) std::round((pos[1] - org[1])*invh);
//   const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
//   const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
//   return std::array<int, 2>{{ix, iy}};
// };