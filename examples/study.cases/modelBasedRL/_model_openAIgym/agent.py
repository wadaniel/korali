#!/usr/bin/env python3

import torch
import os
import sys
import gym
#import pyBulletEnvironments
import math
from HumanoidWrapper import HumanoidWrapper
from AntWrapper import AntWrapper
from model import Net, Hyperparams, NetConfig, predictEnsembleSerial
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import shutil
from mpi4py import MPI
import statistics
import json

#TODO: check state and action dimensions, adapt for OpenAIGym agents!!

sampleInputs = {}
sampleOutputs = {}
surrogateUse = {}

def confidence_range(var, rangeOutputData):
    return 1 - var / (rangeOutputData**2)
    
def confidence_range_2(var, rangeOutputData):
    return math.exp(confidence_range(var, rangeOutputData) - 1)
    
def confidence_var(var, varOutputData):
    return 1 - var / (varOutputData)

class ArgsEnv():

    def __init__(self, dataPoints=0, totalDataPoints=0, previousDataPoints=0, previousTotalDataPoints=0, iterationSurrogate=0, scalers=None, rangeOutputState=None, varOutputState=None, rangeOutputRew=None, varOutputRew=None, dumpBestTrajectory=False, initialRetrainingDataPoints=5000, retrainingDataPoints=1000, keep_retraining=True, policyTestingEpisodes=30, mode="Training", samplesTrained=[], models={}, surrUsage=[], interactionsWithReal=0, interactionsWithSurr=0, totalInteractionsWithEnv=0):
    
        self.dataPoints = dataPoints
        self.totalDataPoints = totalDataPoints
        self.previousDataPoints = previousDataPoints
        self.previousTotalDataPoints = previousTotalDataPoints
        self.iterationSurrogate = iterationSurrogate
        self.scalers = scalers
        self.rangeOutputState = rangeOutputState
        self.varOutputState = varOutputState
        self.rangeOutputRew = rangeOutputRew
        self.varOutputRew = varOutputRew
        self.dumpBestTrajectory = dumpBestTrajectory
        self.initialRetrainingDataPoints = initialRetrainingDataPoints
        self.retrainingDataPoints = retrainingDataPoints
        self.keep_retraining = keep_retraining
        self.policyTestingEpisodes = policyTestingEpisodes
        self.mode = mode
        self.samplesTrained = samplesTrained
        self.models = models
        self.surrUsage = surrUsage
        self.interactionsWithReal = interactionsWithReal
        self.interactionsWithSurr = interactionsWithSurr
        self.totalInteractionsWithEnv = totalInteractionsWithEnv
        self.listInteractionsWithReal = []
        self.listInteractionsWithSurr = []
        self.listSampleIds = []
        self.sampleIdSourcePairs = {}
        
def getDoneAgent(envName):
    #env=Ant-v2
    #env=HalfCheetah-v2
    #env=Hopper-v2
    #env=Humanoid-v2
    #env=HumanoidStandup-v2
    #env=InvertedDoublePendulum-v2
    #env=InvertedPendulum-v2
    #env=Reacher-v2
    #env=Swimmer-v2
    #env=Walker2d-v2
    if envName == "Reacher-v2":
        return False

def dumpTrainingSamples(dirfiles, sampleIdSourcePairs):
    with open(dirfiles["Results"] + "/state.json") as fstate:
        stateJson = json.load(fstate)
    
    inputsInReplay = []
    outputsInReplay = []
    
    print(sampleIdSourcePairs)
    
    experienceReplay = stateJson["Experience Replay"]
    for idx in range(len(experienceReplay)):
        sampleInReplay = experienceReplay[idx]
        if sampleInReplay["Is On Policy"] and idx < len(experienceReplay)-1:
            nextSampleInReplay = experienceReplay[idx+1]
            if nextSampleInReplay["Episode Pos"] != 0:
                # ONLY if real sample:
                if sampleInReplay["Episode Id"] in sampleIdSourcePairs:
                    if sampleInReplay["Episode Pos"] in sampleIdSourcePairs[sampleInReplay["Episode Id"]]:
                        if sampleIdSourcePairs[sampleInReplay["Episode Id"]][sampleInReplay["Episode Pos"]] == "R":
                            inputsInReplay.append([sampleInReplay["Episode Id"]] +  [sampleInReplay["Episode Pos"]] +  sampleInReplay["State"] + sampleInReplay["Action"])
                            difference = np.array(nextSampleInReplay["State"]) - np.array(sampleInReplay["State"])
                            difference = difference.tolist()
                            outputsInReplay.append(difference + [sampleInReplay["Reward"]] + [1.0 - 1.0*sampleInReplay["Reward"]])
                        print(sampleInReplay["Episode Id"], sampleInReplay["Episode Pos"], sampleIdSourcePairs[sampleInReplay["Episode Id"]][sampleInReplay["Episode Pos"]])
                        
                
    print("Size experience replay input / output", len(inputsInReplay), len(outputsInReplay))
                
    dataDumpInp = np.array(inputsInReplay)
    dataDumpOut = np.array(outputsInReplay)
    if dataDumpInp.size > 0:
        dataDump = np.concatenate((dataDumpInp, dataDumpOut), axis=1)
        print(dataDump)
        with open(dirfiles["Dataset Train Korali"], "w") as fwrite:
            np.savetxt(fwrite, dataDump, delimiter=',', fmt='%.5f', header="#traj, step, s0, s1, ..., sn-1, a0, a1, ..., am-1, y0, y1, ..., yk-1, r, done")
            

def dumpTrajectory(trajectoryToDump, dirfiles):
    fname_traj = dirfiles["Best Trajectory"]
    
    dataDumpTrajectory = np.array(trajectoryToDump)
        
    with open(fname_traj, 'w') as fhandle_traj:
        np.savetxt(fhandle_traj, dataDumpTrajectory, delimiter=',', fmt='%.5f')
        
def initGymEnv(envName):

    # Creating environment

    env = gym.make(envName)

    # Handling special cases

    if (envName == 'Humanoid-v2'):
        env = HumanoidWrapper(env)
  
    if (envName == 'HumanoidStandup-v2'):
        env = HumanoidWrapper(env)
  
    if (envName == 'Ant-v2'):
        env = AntWrapper(env)
  
    # Re-wrapping if saving a movie
    if (moviePath != ''):
        env = gym.wrappers.Monitor(env, moviePath, force=True)
        
    # Getting environment variable counts
    stateVariableCount = env.observation_space.shape[0]
    actionVariableCount = env.action_space.shape[0]

    # Generating state variable index list
    stateVariablesIndexes = range(stateVariableCount)

    # Handling Environment-Specific Configuration

    if (envName == 'Ant-v2'):
        stateVariableCount = 27
        stateVariablesIndexes = range(stateVariableCount)
        
    return env, stateVariableCount, actionVariableCount, stateVariablesIndexes

def initEnvironment(e, envName, env, stateVariableCount, actionVariableCount, stateVariablesIndexes, net_config, tags, argsEnv, dirfiles, args, moviePath = ''):

    #net_config.stateVariableCount = stateVariableCount
    #net_config.actionVariableCount = actionVariableCount

    ### Defining problem configuration for openAI Gym environments
    e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
    e["Problem"]["Environment Function"] = lambda x : agent(x, env, net_config, tags, argsEnv, dirfiles)
    e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
    e["Problem"]["Training Reward Threshold"] = math.inf
    e["Problem"]["Policy Testing Episodes"] = args.polTestEp
    e["Problem"]["Testing Frequency"] = args.testFreq
 
    # Defining State Variables

    for i in stateVariablesIndexes:
        e["Variables"][i]["Name"] = "State Variable " + str(i)
        e["Variables"][i]["Type"] = "State"
        e["Variables"][i]["Lower Bound"] = float(env.observation_space.low[i])
        e["Variables"][i]["Upper Bound"] = float(env.observation_space.high[i])
  
    # Defining Action Variables

    for i in range(actionVariableCount):
        e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
        e["Variables"][stateVariableCount + i]["Type"] = "Action"
        e["Variables"][stateVariableCount + i]["Lower Bound"] = float(env.action_space.low[i])
        e["Variables"][stateVariableCount + i]["Upper Bound"] = float(env.action_space.high[i])
        e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = math.sqrt(0.2)

    ### Defining Termination Criteria

    e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = math.inf
 
def agent(s, env, net_config, tags, argsEnv, dirfiles):

    if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
        printStep = True
    else:
        printStep = False
        
    sampleId = s["Sample Id"]
    launchId = s["Launch Id"]
    
    trainingNotTestingMode = (((launchId - sampleId) % argsEnv.policyTestingEpisodes == 0) and (argsEnv.mode == "Training")) and sampleId not in argsEnv.samplesTrained
    
    stateVariableCount = net_config.stateVariableCount
    actionVariableCount = net_config.actionVariableCount
    
    # Coordination Env - Surrogate:
    if trainingNotTestingMode:
    
        argsEnv.samplesTrained.append(sampleId)
        
        if argsEnv.iterationSurrogate == 0:
            if argsEnv.previousDataPoints > argsEnv.initialRetrainingDataPoints:
                # Dump data points to dataset_train_total.csv file for initial training of the ensemble:
                
                #dumpTrainingSamples(dirfiles)
                
                fname_total = dirfiles["Total Dataset Train"]
                shutil.copy(dirfiles["Dataset Train Korali"], dirfiles["Previous Dataset Train Korali"])
                shutil.copy(dirfiles["Dataset Train Korali"], dirfiles["Total Dataset Train"])
                #print(f"[Korali] Written Korali samples to {fname_total}. New samples added by Korali: {dataset_current.shape[0]}")
                fname = dirfiles["Dataset Train Korali"]
                info = "#traj, step, s0, s1, ..., sn-1, a0, a1, ..., am-1, y0, y1, ..., yk-1, r, done"
                with open(fname, "w") as fhandle:
                    fhandle.write(info + "\n")
                           
               # dataset_train_korali is data to train in next iteration, so should calculate ranges and variances of data from past iteration to use it in next (which is data to train ensemble in up to current iteration)
                
                argsEnv.previousDataPoints = 0
                argsEnv.iterationSurrogate += 1
                argsEnv.keep_retraining = True
                
                for r in range(1, net_config.num_procs):
                    net_config.comm.send(argsEnv.keep_retraining, dest=r, tag=tags["tag_keep_retraining"])
                    net_config.comm.send(argsEnv.iterationSurrogate, dest=r, tag=tags["tag_iter_surr"])
                    
        else: # iterationSurrogate = 1, ...
            # in iteration 1, should keep using real environment. In 2, start using surrogate and need everything and so on
            checkDataPoints = argsEnv.initialRetrainingDataPoints if argsEnv.iterationSurrogate == 1 else argsEnv.retrainingDataPoints
            if argsEnv.previousDataPoints > checkDataPoints and argsEnv.keep_retraining:
            
                """
                Changes: instead of reading 'Total Dataset', I read 'Previous Dataset' for scalers and also for ranges !!!
                """
            
                # Concurrently training network with 'dataset_train_korali_prev' and total data seen by ensemble in training is 'dataset_train_total', so calculate scalars to use this model in next iteration based on this data before updating datasets
                inputNumber = stateVariableCount + actionVariableCount
                outputNumberState = stateVariableCount
                outputNumberRew = 1
                usecols_X_train = tuple(range(2, inputNumber+2))
                usecols_y_train_state = tuple(range(inputNumber+2, inputNumber+2+outputNumberState))
                usecols_y_train_rew = (inputNumber+2+outputNumberState)
                X_train_nonscaled_prev = np.genfromtxt(dirfiles["Previous Dataset Train Korali"], delimiter=",", skip_header=1, usecols=usecols_X_train)
                y_train_state_nonscaled_prev = np.genfromtxt(dirfiles["Previous Dataset Train Korali"], delimiter=",", skip_header=1, usecols=usecols_y_train_state)
                y_train_rew_nonscaled_prev = np.genfromtxt(dirfiles["Previous Dataset Train Korali"], delimiter=",", skip_header=1, usecols=usecols_y_train_rew)
                
                #TODO: correct scalers with train_test_split!!
                #TODO: validation with 50-50% split?
                X_train_nonscaled, _, y_train_state_nonscaled, _, y_train_rew_nonscaled, _ = train_test_split(X_train_nonscaled_prev, y_train_state_nonscaled_prev, y_train_rew_nonscaled_prev, train_size=0.8, random_state=1)

                scaler_X_train = StandardScaler().fit(X_train_nonscaled)
                scaler_y_train_state = StandardScaler().fit(y_train_state_nonscaled)
                scaler_y_train_rew = StandardScaler().fit(y_train_rew_nonscaled.reshape(-1, 1))
                argsEnv.scalers = {"scaler_X_train": scaler_X_train, "scaler_y_train_state": scaler_y_train_state, "scaler_y_train_rew": scaler_y_train_rew}
                if scaler_X_train != None:
                    print("[KoraliEnv] Scaler for X_train for next iteration: mean=",scaler_X_train.mean_,"and var=",scaler_X_train.var_,"on",scaler_X_train.n_samples_seen_,"samples seen.")
                if scaler_y_train_state != None:
                    print("[KoraliEnv] Scaler for y_train_state for next iteration: mean=",scaler_y_train_state.mean_,"and var=",scaler_y_train_state.var_,"on",scaler_y_train_state.n_samples_seen_,"samples seen.")
                if scaler_y_train_rew != None:
                    print("[KoraliEnv] Scaler for y_train_rew for next iteration: mean=",scaler_y_train_rew.mean_,"and var=",scaler_y_train_rew.var_,"on",scaler_y_train_rew.n_samples_seen_,"samples seen.")
                
                #TODO: if ranges from previous or total dataset
                if net_config.scaleInputAndOutput:
                    outputDataState = scaler_y_train_state.transform(np.genfromtxt(dirfiles["Previous Dataset Train Korali"], delimiter=",", skip_header=1, usecols=usecols_y_train_state))
                    outputDataRew = scaler_y_train_rew.transform(np.genfromtxt(dirfiles["Previous Dataset Train Korali"], delimiter=",", skip_header=1, usecols=usecols_y_train_rew).reshape(-1, 1))
                    argsEnv.rangeOutputState = np.absolute(np.ptp(outputDataState, axis=0))
                    argsEnv.varOutputState = np.var(outputDataState, axis=0)
                    argsEnv.rangeOutputRew = np.absolute(np.ptp(outputDataRew, axis=0))[0]
                    argsEnv.varOutputRew = np.var(outputDataRew, axis=0)[0]
                    
                else:
                    outputDataState = np.genfromtxt(dirfiles["Previous Dataset Train Korali"], delimiter=",", skip_header=1, usecols=usecols_y_train_state)
                    outputDataRew = np.genfromtxt(dirfiles["Previous Dataset Train Korali"], delimiter=",", skip_header=1, usecols=usecols_y_train_rew)
                    argsEnv.rangeOutputState = np.absolute(np.ptp(outputDataState, axis=0))
                    argsEnv.varOutputState = np.var(outputDataState, axis=0)
                    argsEnv.rangeOutputRew = np.absolute(np.ptp(outputDataRew, axis=0))
                    argsEnv.varOutputRew = np.var(outputDataRew, axis=0)
                
                # Wait for ranks telling retrained model is ready and load it
                textPrint = "";
                print("[Korali] Waiting for the other ranks to finish re-training the surrogate model ...")
                for r in range(1, net_config.num_procs):
                    textPrint += net_config.comm.recv(source=r, tag=tags["tag_retrained_ready"])
                print("[Korali] Done")
                print(textPrint)
        
                for r in range(1, net_config.num_procs):
                    argsEnv.models[r] = Net(net_config.hyperparams.hidden_size, net_config.alphaDropout)
                    PATH_prev = dirfiles[f"Previous Model {r}"]
                    checkpoint = torch.load(PATH_prev)
                    argsEnv.models[r].load_state_dict(checkpoint['state_dict'])
                    argsEnv.models[r].eval()
                    #print("[Korali] Models object:", argsEnv.models)
                    
                dataset_previous = np.genfromtxt(dirfiles["Total Dataset Train"], delimiter=",", skip_header=1)
                dataset_current = np.genfromtxt(dirfiles["Dataset Train Korali"], delimiter=",", skip_header=1)
                fname_total = dirfiles["Total Dataset Train"]
                np.savetxt(fname_total, np.concatenate((dataset_previous, dataset_current), axis=0), delimiter=",", fmt='%.5f', header="#traj, step, s0, s1, ..., sn-1, a0, a1, ..., am-1, y0, y1, ..., yk-1, r, done")
                
                shutil.copy(dirfiles["Dataset Train Korali"], dirfiles["Previous Dataset Train Korali"])
                #fname = dirfiles["Dataset Train Korali"]
                #info = "#traj, step, s0, s1, ..., sn-1, a0, a1, ..., am-1, y0, y1, ..., yk-1, r, done"
                #with open(fname, "w") as fhandle:
                #    fhandle.write(info + "\n")
                dumpTrainingSamples(dirfiles, argsEnv.sampleIdSourcePairs)

                argsEnv.previousDataPoints = 0
                argsEnv.iterationSurrogate += 1
                argsEnv.keep_retraining = True

                for r in range(1, net_config.num_procs):
                    net_config.comm.send(argsEnv.keep_retraining, dest=r, tag=tags["tag_keep_retraining"])
                    net_config.comm.send(argsEnv.iterationSurrogate, dest=r, tag=tags["tag_iter_surr"])

    
    trajectoryToDump = []
    
    confidences = []
    rewardsSurr = []
    
    #TODO: give seed to OpenAIGym env's?
    #seed = sampleId * 1024 + launchId
    #rng = np.random.default_rng(seed)
    
    if argsEnv.iterationSurrogate >= 2:
        print(f"[Env] SampleId={sampleId} - LaunchId={launchId} - Previous Data Points={argsEnv.previousDataPoints} - Iteration Surrogate={argsEnv.iterationSurrogate}")
  
    s["State"] = env.reset().tolist()
    done = False
    step = 0
    stepReal = 0
    usedSurr = 0
    
    sampleInputs[sampleId] = []
    sampleOutputs[sampleId] = []
    surrogateUse[sampleId] = []
    
    confidenceRange = 1  #dummy
    confidenceVar = 1  #dummy

    # Storage for cumulative reward
    cumulativeReward = 0.0

    overSteps = 0

    while not done and step < 1000:
    
        previousStep = step
        previousState = s["State"] # list

        # Getting new action
        s.update()

        # Printing step information
        if (printStep):  print('[Korali] Frame ' + str(step), end = '')
  
        # Performing the action
        action = s["Action"] # list
        
        if argsEnv.iterationSurrogate < 2 or not trainingNotTestingMode or argsEnv.mode == "TestingReal":
            dump = False
            if argsEnv.iterationSurrogate < 2 and trainingNotTestingMode:
                dump = True
            state, reward, done, difference = agentReal(s, env, previousStep, previousState, action, net_config, dump)
            
            if trainingNotTestingMode:
                stepReal += 1
                argsEnv.interactionsWithReal += 1
                if sampleId not in argsEnv.sampleIdSourcePairs:
                    argsEnv.sampleIdSourcePairs[sampleId] = {step:"R"}
                else:
                    argsEnv.sampleIdSourcePairs[sampleId][step] = "R"
                    
        else:
            state, confidenceRange, confidenceVar, reward, done, difference = agentSurr(s, envName, previousState, action, net_config, argsEnv.scalers, argsEnv.models, argsEnv.rangeOutputState, argsEnv.varOutputState, argsEnv.rangeOutputRew, argsEnv.varOutputRew)
            
            rewardsSurr.append(reward)
            
            if confidenceRange < net_config.hyperparams.confidence:
                #TODO: reset env state from gym, and time and so on, and step, etc!!
                #cart.u = np.array([previousState[0], previousState[1], previousState[2], previousState[3]]) #was np.asarray
                #cart.t = step*cart.dt
                #cart.step = step
                state, reward, done, difference = agentReal(s, env, previousStep, previousState, action, net_config, dump=False)
                #s["SurrogateUse"].append(0)
                
                if trainingNotTestingMode:
                    stepReal += 1
                    surrogateUse[sampleId].append(0)
                    argsEnv.interactionsWithReal += 1
                    if sampleId not in argsEnv.sampleIdSourcePairs:
                        argsEnv.sampleIdSourcePairs[sampleId] = {step:"R"}
                    else:
                        argsEnv.sampleIdSourcePairs[sampleId][step] = "R"
            else:
                if trainingNotTestingMode:
                    usedSurr += 1
                    surrogateUse[sampleId].append(1)
                    argsEnv.interactionsWithSurr += 1
                    if sampleId not in argsEnv.sampleIdSourcePairs:
                        argsEnv.sampleIdSourcePairs[sampleId] = {step:"S"}
                    else:
                        argsEnv.sampleIdSourcePairs[sampleId][step] = "S"

        # Getting Reward
        s["Reward"] = reward

        # Printing step information
        #if (printStep):  print(' - State: ' + str(state) + ' - Action: ' + str(action))
        cumulativeReward = cumulativeReward + reward
        if (printStep):  print(' - Cumulative Reward: ' + str(cumulativeReward))
   
        # Storing New State
        s["State"] = state.tolist()

        # Advancing step counter
        step = step + 1
        argsEnv.totalInteractionsWithEnv += 1
        
        if argsEnv.dumpBestTrajectory and argsEnv.mode == "TestingReal" and sampleId == 1:
            trajectoryToDump.append([step] + previousState + s["Action"] + state + [reward])
        confidences.append(confidenceRange)

    if trainingNotTestingMode:
        argsEnv.previousDataPoints = argsEnv.previousDataPoints + stepReal
        argsEnv.previousTotalDataPoints = argsEnv.previousTotalDataPoints + stepReal
        argsEnv.listInteractionsWithReal.append(argsEnv.interactionsWithReal)
        argsEnv.listInteractionsWithSurr.append(argsEnv.interactionsWithSurr)
        argsEnv.listSampleIds.append(sampleId)

    # Setting termination status
    if (done):
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"
        
    if trainingNotTestingMode:
    
        if argsEnv.iterationSurrogate < 2:
    
            fname = dirfiles["Dataset Train Korali"]
            #info = "s0, s1, s2, s3, a, y0, y1, y2, y3, r"
        
            dataDumpInp = np.array(sampleInputs[sampleId])
            dataDumpOut = np.array(sampleOutputs[sampleId])
            if dataDumpInp.size > 0:
                dataDump = np.concatenate((dataDumpInp, dataDumpOut), axis=1)

                if os.path.exists(fname):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' # make a new file if not

                with open(fname, append_write) as fhandle:
                    np.savetxt(fhandle, dataDump, delimiter=',', fmt='%.5f')
        
        fname_usage = dirfiles["Model Usage"]
        
        dataDumpUsage = np.array(surrogateUse[sampleId])
        argsEnv.surrUsage.append(surrogateUse[sampleId])
        
        if os.path.exists(fname_usage): #and sampleId > 1:
            append_write_us = 'a' # append if already exists
        else:
            append_write_us = 'w' # make a new file if not
            
        with open(fname_usage, append_write_us) as fhandle_us:
            np.savetxt(fhandle_us, dataDumpUsage, delimiter=',', fmt='%1i')
            
        #if argsEnv.dumpBestTrajectory:
        #    dumpTrajectory(trajectoryToDump, net_config.dt_string)
            
        if argsEnv.iterationSurrogate >= 2:
        
            stdevConfidences = statistics.stdev(confidences)
            worstConfidence = min(confidences)
            bestConfidence = max(confidences)
            meanConfidence = statistics.mean(confidences)
            
            stdevRew = statistics.stdev(rewardsSurr)
            minRew = min(rewardsSurr)
            maxRew = max(rewardsSurr)
            meanRew = statistics.mean(rewardsSurr)
        
            surrPercent = usedSurr / step
        
            print(f"[Env]   + Confidence [worst / mean / best / stdev] = [{worstConfidence:.5f} / {meanConfidence:.5f} / {bestConfidence:.5f} / {stdevConfidences:.5f}] - Surr Usage = {surrPercent:.5f}")
            print(f"[Env]   + Reward [min / mean / max / stdev] = [{minRew:.5f} / {meanRew:.5f} / {maxRew:.5f} / {stdevRew:.5f}]")
    
    if argsEnv.dumpBestTrajectory and argsEnv.mode == "TestingReal" and sampleId == 1:
        dumpTrajectory(trajectoryToDump, dirfiles)

def agentReal(s, env, previousStep, previousState, action, net_config, dump=True):
    
    sampleId = s["Sample Id"]
    
    if dump:
        sampleInputs[sampleId].append([sampleId] + [previousStep] + previousState + action)
        
    state, reward, done, _ = env.step(action)
    
    difference = state - np.array(previousState)
    #difference[0] = state[0] - previousState[0]
    #difference[1] = state[1] - previousState[1]
    #difference[2] = state[2] - previousState[2]
    #difference[3] = state[3] - previousState[3]
    difference = difference.tolist()
    
    if dump:
        sampleOutputs[sampleId].append(difference + [reward] + [1.0 - 1.0*reward])

    return state, reward, done, difference


def agentSurr(s, envName, previousState, action, net_config, scalers, models, rangeOutputState, varOutputState, rangeOutputRew, varOutputRew):
    
    with torch.no_grad():

        # Predict next state, advance cartpole:
        if net_config.hyperparams.scaleData:
            inp = torch.Tensor(scalers["scaler_X_train"].transform([previousState + action]))
        else:
            inp = torch.Tensor([previousState + action])
        
        prediction1_mu, prediction1_var, prediction2_mu, prediction2_var = predictEnsembleSerial(net_config, inp, models)
        
        if net_config.scaleInputAndOutput:
            pred1_mu = scalers["scaler_y_train_state"].inverse_transform(prediction1_mu[0].tolist())  # vector
            pred2_mu = scalers["scaler_y_train_rew"].inverse_transform(prediction2_mu[0])[0]
        else:
            pred1_mu = prediction1_mu[0]      # vector
            pred2_mu = prediction2_mu[0][0]   # scalar
        
        prediction1_mu = prediction1_mu[0]      # vector
        prediction2_mu = prediction2_mu[0][0]   # scalar
        
        # Store new state:
        state = pred1_mu.numpy() - np.array(previousState)
        #state[0] = pred1_mu[0] + previousState[0]
        #state[1] = pred1_mu[1] + previousState[1]
        #state[2] = pred1_mu[2] + previousState[2]
        #state[3] = pred1_mu[3] + previousState[3]
        state = state.tolist()

        difference = pred1_mu.tolist()
        prediction1_var = prediction1_var[0].tolist()
        variances = prediction1_var
        prediction2_var = prediction2_var[0][0]
        
        reward = pred2_mu.item()

        done = getDoneAgent(envName)
        #done = abs(state[0])>x_threshold or abs(state[2])>th_threshold
        
        confidenceInStateRange = 0
        for i in range(len(variances)):
            confidenceInStateRange += confidence_range_2(variances[i], rangeOutputState[i])
        confidenceInStateRange /= len(variances)
        
        confidenceInStateVar = 0
        for i in range(len(variances)):
            confidenceInStateVar += confidence_var(variances[i], varOutputState[i])
        confidenceInStateVar /= len(variances)
        
        confidenceInRewRange = confidence_range_2(prediction2_var, rangeOutputRew)
        
        confidenceInRewVar = confidence_var(prediction2_var, varOutputRew)
        
        confidenceRange = np.average([confidenceInStateRange, confidenceInRewRange])
        confidenceVar = np.average([confidenceInStateVar, confidenceInRewVar])

    return state, confidenceRange, confidenceVar, reward, done, difference


