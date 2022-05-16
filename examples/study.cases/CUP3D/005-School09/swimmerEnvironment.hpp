//  Korali environment for CubismUP-3D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include <filesystem>
#include <iostream>
#include <fstream>

#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"
#include <Cubism/ArgumentParser.h>

#define STATES 35
#define AGENTS 9
#define ACTIONS 8

#if modelDIM == 2
    std::string OPTIONS = "-bpdx 2 -bpdy 2 -levelMax 7 -levelStart 5 -Rtol 5.0 -Ctol 0.1 -extent 2.0 -CFL 0.7 -poissonTol 1e-5 -poissonTolRel 1e-2 -bMeanConstraint 0 -tdump 0 -nu 0.00001 -poissonSolver cuda_iterative";
    std::vector<std::vector<double>> initialPositions{{
          {0.60, 1.00, 1.00},
          {0.90, 0.90, 1.00},
          {0.90, 1.10, 1.00},
          {1.20, 0.80, 1.00},
          {1.20, 1.00, 1.00},
          {1.20, 1.20, 1.00},
          {1.50, 0.90, 1.00},
          {1.50, 1.10, 1.00},
          {1.80, 1.00, 1.00}
     }};
     std::vector<std::vector<double>> actual_initialPositions;
#else
    std::string OPTIONS = " -bpdx 2 -bpdy 2 -bpdz 1 -extentx 2.0 -levelMax 7 -levelStart 5 -Rtol 5.0 -Ctol 0.1 -tdump 0 -CFL 0.7 -lambda 1e6 -nu 0.00001 -poissonTol 1e-5 -poissonTolRel 1e-2 -bMeanConstraint 1 ";
    std::vector<std::vector<double>> initialPositions{{
          {0.60, 1.00, 1.00},
          {0.90, 0.90, 1.00},
          {0.90, 1.10, 1.00},
          {1.20, 0.80, 1.00},
          {1.20, 1.00, 1.00},
          {1.20, 1.20, 1.00},
          {1.50, 0.90, 1.00},
          {1.50, 1.10, 1.00},
          {1.80, 1.00, 1.00}
     }};
     std::vector<std::vector<double>> actual_initialPositions;
    using namespace cubismup3d;
#endif

std::mt19937 _randomGenerator;

void runEnvironment(korali::Sample &s);

namespace
{

Simulation * initializeEnvironment(korali::Sample &s)
{
    MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
    int rank;
    MPI_Comm_rank(comm,&rank);

    // Argument string to inititialize Simulation
    #if modelDIM == 2
        std::string argumentString = "CUP-RL " + OPTIONS + " -shapes ";
        std::string AGENT = " \n\
        stefanfish L=0.2 T=1 bFixed=1 ";
    #else
        std::string argumentString = "CUP-RL " + OPTIONS + " -factory-content ";
        std::string AGENT = " \n\
        StefanFish L=0.2 T=1 bFixToPlanar=1 bFixFrameOfRef=1 ";
    #endif

    // Set initial position for all prey
    for( int a = 0; a < AGENTS; a++ )
    {
        std::vector<double> initialData = initialPositions[a];
        if ( s["Mode"] == "Training" ) // During training, add noise to initial positions
        {
            if (rank == 0) // only rank 0 samples initial data and broadcasts it
            {
                std::uniform_real_distribution<double> disX(-0.05, 0.05);
                std::uniform_real_distribution<double> disY(-0.05, 0.05);
                std::uniform_real_distribution<double> disZ(-0.05, 0.05);
                initialData[0] += disX(_randomGenerator);
                initialData[1] += disY(_randomGenerator);
                //initialData[2] += disZ(_randomGenerator);
            }
            MPI_Bcast(initialData.data(), 3, MPI_DOUBLE, 0, comm);
        }
        actual_initialPositions.push_back(initialData);
        // Append agent to argument string
        argumentString = argumentString + AGENT + " xpos=" + std::to_string(initialData[0]) 
                                                + " ypos=" + std::to_string(initialData[1]);
        #if modelDIM == 3
            argumentString = argumentString     + " zpos=" + std::to_string(initialData[2])
                                                + " heightProfile=danio" + " widthProfile=stefan";
        #endif
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

    #if modelDIM == 2
        Simulation *_environment = new Simulation(argv.size() - 1, argv.data(), comm);
        _environment->init();

        // Establishing environment's dump frequency
        _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();
        
    #else
        ArgumentParser parser(argv.size()-1, argv.data());
        Simulation *_environment = new Simulation(comm, parser);

        // Establishing environment's dump frequency
        _environment->sim.saveTime = s["Custom Settings"]["Dump Frequency"].get<double>();
    #endif

    return _environment;
}

bool isTerminal(StefanFish *agent, const SimulationData & sim, const int agentID, const size_t step)
{
    const size_t maxSteps = 1000; // Max steps before truncation

    if (step >= maxSteps) return true;

    if (sim.bCollision) return true;

    const double xMin = 0.1;
    const double xMax = 1.9;
    const double yMin = 0.1;
    const double yMax = 1.9;
    const double zMin = 0.1;
    const double zMax = 1.9;
    #if modelDIM == 2
        const double X = agent->center[0];
        const double Y = agent->center[1];
        const double Z = 1.0;
    #endif
    #if modelDIM == 3
        const double X = agent->absPos[0];
        const double Y = agent->absPos[1];
        const double Z = agent->absPos[2];
    #endif
    if (X < xMin) return true;
    if (X > xMax) return true;
    if (Y < yMin) return true;
    if (Y > yMax) return true;
    if (Z < zMin) return true;
    if (Z > zMax) return true;
    return false;
}

double getReward(std::vector<StefanFish *> & agents, const SimulationData & sim, const int agentID)
{
    //Check if this fish collided and return penalty only if it did.
    if (sim.bCollision) 
    {
        for (size_t i = 0; i < sim.bCollisionID.size(); i++)
            if (sim.bCollisionID[i] == agentID) return -10.0;
    }

    // Return penalty if fish exited the domain
    #if modelDIM == 2
    const double X = agents[agentID]->center[0];
    const double Y = agents[agentID]->center[1];
    const double Z = 1.0;
    #else
    const double X = agents[agentID]->absPos[0];
    const double Y = agents[agentID]->absPos[1];
    const double Z = agents[agentID]->absPos[2];
    #endif
    const double xMin = 0.1;
    const double xMax = 1.9;
    const double yMin = 0.1;
    const double yMax = 1.9;
    const double zMin = 0.1;
    const double zMax = 1.9;
    if (X < xMin) return -10;
    if (X > xMax) return -10;
    if (Y < yMin) return -10;
    if (Y > yMax) return -10;
    if (Z < zMin) return -10;
    if (Z > zMax) return -10;

    // Reward swimming efficiency is fish did not collide and is in the domain
    return agents[agentID]->EffPDefBnd;
}

std::vector<double> getState(std::vector<StefanFish *> & agents, const SimulationData & sim, const int agentID)
{
    std::vector<double> S;
    #if modelDIM == 3
        std::vector<double> s_base = agents[agentID]->state();
        const double X = agents[agentID]->absPos[0];
        const double Y = agents[agentID]->absPos[1];
        const double Z = agents[agentID]->absPos[2];
    #else
        std::vector<double> s_base = agents[agentID]->state(actual_initialPositions[agentID]);
        const double X = agents[agentID]->center[0];
        const double Y = agents[agentID]->center[1];
        const double Z = 0;
    #endif
    for (size_t i = 0 ; i < s_base.size() ; i++)
            S.push_back(s_base[i]);

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> d;
    for (int i = 0 ; i < (int)agents.size(); i++)
    {
        if (i == agentID) continue;
        #if modelDIM == 2
            const double xfish = agents[i]->center[0];
            const double yfish = agents[i]->center[1];
            const double zfish = 0;
        #else
            const double xfish = agents[i]->absPos[0];
            const double yfish = agents[i]->absPos[1];
            const double zfish = agents[i]->absPos[2];
        #endif
        x.push_back(xfish-X);
        y.push_back(yfish-Y);
        z.push_back(zfish-Z);
        d.push_back(std::sqrt((xfish-X)*(xfish-X)+(yfish-Y)*(yfish-Y)+(zfish-Z)*(zfish-Z)));
    }
    std::vector<int> indices(d.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int A, int B) -> bool {return d[A] < d[B];});
    
    for (int i = 0 ; i < 3; i++)//distance from three closest fish saved
    {
        S.push_back(x[indices[i]]);
        S.push_back(y[indices[i]]);
        S.push_back(z[indices[i]]);
    }
    if (S.size() != STATES)
    {
       std::cout << "wrong state size " << S.size() << "->" << STATES << std::endl;
       abort();
    }
    return S;
}

void takeAction(StefanFish *agent, const SimulationData & sim, const int agentID, 
                const std::vector<double> & action, const double l_tnext)
{
    #if modelDIM == 3
        //eight actions defined for compatibility with 3D motion, only two used for planar motion
        auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( agent->myFish );
        cFish->action_curvature(sim.time,l_tnext,action[0]);
        cFish->action_period   (sim.time,l_tnext,action[1]);
    #else
        agent->act(l_tnext,action);
    #endif
}

#ifndef EVALUATION
void setupRL(korali::Experiment & e)
{
    e["Problem"]["Agents Per Environment"] = AGENTS;
    e["Problem"]["Policies Per Environment"] = 1;

    size_t curVariable = 0;
    for (; curVariable < STATES; curVariable++)
    {
        e["Variables"][curVariable]["Name"] = std::string("State") + std::to_string(curVariable);
        e["Variables"][curVariable]["Type"] = "State";
    }

    e["Variables"][curVariable]["Name"] = "Curvature";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -1.0;
    e["Variables"][curVariable]["Upper Bound"] = +1.0;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Swimming Period";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.25;
    e["Variables"][curVariable]["Upper Bound"] = +0.25;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 0";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.2;
    e["Variables"][curVariable]["Upper Bound"] = +0.2;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.05;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 1";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.2;
    e["Variables"][curVariable]["Upper Bound"] = +0.2;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.05;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 2";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.2;
    e["Variables"][curVariable]["Upper Bound"] = +0.2;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.05;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 3";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.5;
    e["Variables"][curVariable]["Upper Bound"] = +0.5;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.05;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 4";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.5;
    e["Variables"][curVariable]["Upper Bound"] = +0.5;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.05;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 5";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.2;
    e["Variables"][curVariable]["Upper Bound"] = +0.2;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.05;

    e["Solver"]["Experiences Between Policy Updates"] = 1;
    e["Solver"]["Learning Rate"] = 1e-4;
    e["Solver"]["Discount Factor"] = 0.95;
    e["Solver"]["Mini Batch"]["Size"] =  128;
 
    // Configuration of replay memory
    e["Solver"]["Experience Replay"]["Start Size"] = 1024;
    e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
    e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
    e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
    e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
    e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

    // Neural Network
    e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
    e["Solver"]["Neural Network"]["Optimizer"] = "Adam";
 
    e["Solver"]["L2 Regularization"]["Enabled"] = true;
    e["Solver"]["L2 Regularization"]["Importance"] = 1.0;
 
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64;
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";
 
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 64;
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Linear";
    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Output Channels"] = 64;
    e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Activation";
    e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"] = "Elementwise/Tanh";
}
#endif
}
