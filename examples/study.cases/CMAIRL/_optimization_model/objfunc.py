#!/usr/bin/env python
import numpy as np
import sys
import csv
import korali

sys.path.append('./_rl_model/_model')
from env import *
from evalenv import *


# rl cartpole with unknown reward
def rl_cartpole_vracer(p):

    k = korali.Engine()
    e = korali.Experiment()

    # environment with parametrized reward function
    target = p["Parameters"][0]
    envp = lambda s : env(s,target)

    ### Defining the Cartpole problem's configuration

    e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
    e["Problem"]["Environment Function"] = envp
    e["Problem"]["Training Reward Threshold"] = 490
    e["Problem"]["Policy Testing Episodes"] = 10
    e["Problem"]["Actions Between Policy Updates"] = 5

    e["Variables"][0]["Name"] = "Cart Position"
    e["Variables"][0]["Type"] = "State"

    e["Variables"][1]["Name"] = "Cart Velocity"
    e["Variables"][1]["Type"] = "State"

    e["Variables"][2]["Name"] = "Pole Angle"
    e["Variables"][2]["Type"] = "State"

    e["Variables"][3]["Name"] = "Pole Angular Velocity"
    e["Variables"][3]["Type"] = "State"

    e["Variables"][4]["Name"] = "Force"
    e["Variables"][4]["Type"] = "Action"
    e["Variables"][4]["Lower Bound"] = -10.0
    e["Variables"][4]["Upper Bound"] = +10.0

    ## Defining Agent Configuration 

    e["Solver"]["Type"] = "Agent / Continuous / VRACER"
    e["Solver"]["Experiences Between Policy Updates"] = 5
    e["Solver"]["Cache Persistence"] = 500

    e["Solver"]["Refer"]["Target Off Policy Fraction"] = 0.1
    e["Solver"]["Refer"]["Cutoff Scale"] = 4.0

    ## Defining the configuration of replay memory

    e["Solver"]["Experience Replay"]["Start Size"] = 1000
    e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

    ## Defining Neural Network Configuration for Policy and Critic into Critic Container

    e["Solver"]["Critic"]["Discount Factor"] = 0.99
    e["Solver"]["Critic"]["Learning Rate"] = 1e-4
    e["Solver"]["Critic"]["Mini Batch Size"] = 256

    e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
    e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

    e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
    e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 128
    e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

    e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
    e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 128
    e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

    e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
    e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"

    ## Defining Termination Criteria

    e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490
    e["Solver"]["Termination Criteria"]["Max Generations"] = 2500

    ## Setting file output configuration

    e["File Output"]["Enabled"] = False
    e["Console Output"]["Verbosity"] = "Silent"

    ### Running Experiment
    
    k.run(e)

    ### Evaluate Policy
    
    e["Solver"]["Mode"] = "Testing"
    e["Solver"]["Testing"]["Sample Ids"] = [0]
    e["Problem"]["Policy Testing Episodes"] = 1
    e["Problem"]["Environment Function"] = evalenv

    k.run(e)

    print("[Korali] Finished testing.")
    suml2 = e["Solver"]["Testing"]["Average Reward"]
   
    p["F(x)"] = -suml2 # maximize
