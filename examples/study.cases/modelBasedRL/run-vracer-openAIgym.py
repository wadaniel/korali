#!/usr/bin/env python3
import os
import sys
import shutil
import json
import warnings
import argparse
from datetime import datetime
from mpi4py import MPI

sys.path.append('_model_openAIgym')
from agent import *
from model import Net, Hyperparams, NetConfig, trainEnsemble, predictEnsembleSerial, GaussianNLLLoss, VisualizationNetResults

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adabelief_pytorch import AdaBelief
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument(-"-env", help="Specifies which environment to run.", required=True)
parser.add_argument("--dis", help="Sampling Distribution.", required=True)
parser.add_argument("--l2", help="L2 Regularization.", required=False, type=float, default = 0.)
parser.add_argument("--opt", help="Off Policy Target.", required=False, type=float, default = 0.1)
parser.add_argument("--lrRL", help="Learning Rate.", required=False, type=float, default = 0.0001)
parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate surrogate nets")
parser.add_argument("--batch", type=int, default=8, help="Batch size")
parser.add_argument("--epoch", type=int, default=100, help="Total number of epochs to train the surrogate nets")
parser.add_argument("--hid", nargs="+", type=int, help="Specify integers for list of hidden sizes, starting ith input size to first hidden layer and finishing with output size of last layer")
parser.add_argument("--ws", type=float, default=4.0, help="Weight parameter for GaussianNLLLoss of reward (scalar)")
parser.add_argument("--conf", type=float, default=0.9500, help="Confidence Hyperparameter")
parser.add_argument("--m", type=str, default="", help="Prefix results and directories")
parser.add_argument("--dumpBestTrajectory", action='store_true', help="Dump best trajectory")
parser.add_argument("--trRewTh", type=int, default=495, help="trainingRewardThreshold")
parser.add_argument("--tarAvRew", type=int, default=495, help="targetAverageReward")
parser.add_argument("--maxGen", type=int, default=1000, help="maximumGenerations")
parser.add_argument("--epPerGen", type=int, default=1, help="episodesPerGeneration")
parser.add_argument("--testFreq", type=int, default=100, help="testingFrequency")
parser.add_argument("--polTestEp", type=int, default=20, help="policyTestingEpisodes")
parser.add_argument("--expBetPolUp", type=float, default=1, help="Experiences Between Policy Updates")
parser.add_argument("--maxPolUp", type=float, default=0, help="Maximum Policy Updates")
parser.add_argument("--maxSize", type=float, default=262144, help="Maximum size of memory for experience replay")
parser.add_argument("--launchNum", type=int, default=1, help="Number of times to run vracer.")
parser.add_argument("--iniRetrain", type=int, default=65536, help="Initial retraining sample data points.")
parser.add_argument("--retrain", type=int, default=50000, help="Retraining sample data points.")
args = parser.parse_args()
print(args)

comm = MPI.COMM_WORLD   # define communicator for the solver for surrogate model parallel training and predicting (Korali in serial mode)
rank = comm.Get_rank()
num_procs = comm.Get_size()
tags = {"tag_keep_retraining": 11, "tag_iter_surr": 12, "tag_retrained_ready": 13}

now = datetime.now()
if args.m == "":
    args.m = "Results/" + now.strftime("%Y%m%d%H%M%S") + "/"
else:
    args.m = "Results/" + args.env + "_" + args.m  + "/"

if rank == 0:
    if not os.path.exists("Results/"): os.makedirs("Results/")
    if not os.path.exists(args.m): os.makedirs(args.m)
    if not os.path.exists(args.m + "Models/"): os.makedirs(args.m + "Models/")
    if not os.path.exists(args.m + "Visualization/"): os.makedirs(args.m + "Visualization/")

dirfiles = {"Results Launches": args.m + "{:.5f}".format(args.conf) + "_"  + f"_results.csv"}

if rank == 0:
    fname = dirfiles["Results Launches"]
    infores = "#bestTestingEpisodeId,averageTrainingReward,finalGenerationRes,previousDataUsage,usagesurrogateModel,averageTestReward,numberInteractionsWithReal,numberInteractionsWithSurr,totalNumberInteractionsWithEnv,interactionsRealWarmUp,ratioExperiences,policyUpdateCount,iterationsUsed"
    with open(fname, "w") as fhandle:
        fhandle.write(infores+"\n")
        
import korali
        
for launch in range(args.launchNum):

    comm.Barrier()

    # Torch random seed
    torch.manual_seed(rank*args.launchNum + launch)

    dt_string = "L" + str(launch) + "_" + "{:.2f}".format(args.conf) + "_"
    
    dis_dir = args.dis.replace(" ","_")
    resultFolder = 'result_vracer_' + args.env + '_' + dis_dir + '_' + str(args.lr) + '_' + str(args.opt) + '_' + str(args.l2) + '/'

    dirfiles["Interactions"] = args.m + dt_string + f"interactions.csv"
    dirfiles["Real Testing Rewards"] = args.m + dt_string + f"real_rewards.csv"
    dirfiles["Dataset Train Korali"] = args.m + dt_string + f"dataset_train_korali.csv"
    dirfiles["Previous Dataset Train Korali"] = args.m + dt_string + f"dataset_train_korali_prev.csv"
    dirfiles["Total Dataset Train"] = args.m + dt_string + f"dataset_train_total.csv"
    dirfiles["Model Usage"] = args.m + dt_string + f"model_usage.csv"
    dirfiles["Results"] = args.m + "_" + dt_string + resultFolder
    dirfiles["Best Trajectory"] = args.m + "_" + dt_string + "best_trajectory.csv"
    dirfiles["Experience Replay"] = args.m + dt_string + "experience_replay.csv"
    for r in range(1, num_procs):
        dirfiles[f"Current Model {r}"] = args.m + "Models/" + dt_string + f"Rank{r}_of_{num_procs}_surnet_curr.pth"
        dirfiles[f"Previous Model {r}"] = args.m + "Models/" + dt_string + f"Rank{r}_of_{num_procs}_surnet_prev.pth"
        dirfiles[f"Checkpoint Model {r}"] = args.m + "Models/" + dt_string + f"Rank{r}_of_{num_procs}_surnet_checkpoint.pth"
        dirfiles[f"Visualization {r}"] = args.m + "Visualization/" + dt_string + f"Rank{r}_of_{num_procs}_"

    if rank == 0:
        print("\n\n")
        print(f"\n************** {args.env} Surrogate - Launch={launch}/{args.launchNum} - " + dt_string + " **************\n")

    trainingRewardThreshold = args.trRewTh
    targetAverageReward = args.tarAvRew
    episodesPerGeneration = args.epPerGen #2
    policyTestingEpisodes = args.polTestEp
    dumpBestTrajectory = args.dumpBestTrajectory
    testingFrequency = args.testFreq
    experiencesBetweenPolicyUpdates = args.expBetPolUp
    maxPolicyUpdates = args.maxPolUp
    maxSize = args.maxSize
    startSize = 2 * args.iniRetrain

    # Initialize the hyperparameters of the problem
    scale = True
    scaleIO = True
    hyperparams = Hyperparams(lr=args.lr, batch_size=args.batch, epoch_number=args.epoch, hidden_size=args.hid, weight_loss_scalar=args.ws, shuffle=True, confidence=args.conf, scaleData=scale, patience=8, lr_reduction_factor=0.2)

    if rank == 0:
        print(f"[Korali] Maximum Number of Generations = {args.maxGen}")
        print(f"[Korali] Initial number of data points to retrain = {args.iniRetrain}")
        print(f"[Korali] Number of data points to retrain = {args.retrain}")
        print(f"[Korali] Learning Rate Korali NN = {args.lrRL}")
        print(f"[Korali] Training Reward Threshold = {trainingRewardThreshold}")
        print(f"[Korali] Target Average Reward = {targetAverageReward}")
        print(f"[Korali] EpisodesPerGeneration = {episodesPerGeneration}")
        print(f"[Korali] Policy Testing Episodes = {policyTestingEpisodes}")
        print(f"[Korali] Dump Best Trajectory = {dumpBestTrajectory}")
        print(f"[Ensemble] Learning Rate = {hyperparams.lr}")
        print(f"[Ensemble] Batch Size = {hyperparams.batch_size}")
        print(f"[Ensemble] Epoch Number = {hyperparams.epoch_number}")
        print(f"[Ensemble] Hidden Size = {hyperparams.hidden_size}")
        print(f"[Ensemble] Weight Scalar = {hyperparams.weight_loss_scalar}")
        print(f"[Ensemble] Shuffle = {hyperparams.shuffle}")
        print(f"[Ensemble] Confidence = {hyperparams.confidence}")
        print(f"[Ensemble] Scale Dataset = {hyperparams.scaleData}")
        print(f"[Ensemble] Size of the communicator is {num_procs} ranks")
        print(dirfiles)
        
    # Init Gym env
    env, stateVariableCount, actionVariableCount, stateVariablesIndexes = initGymEnv(args.env)
    inputNumber = stateVariableCount + actionVariableCount
    outputNumberState = stateVariableCount

    # Loss functions
    loss_fn_state = GaussianNLLLoss()
    loss_fn_rew = GaussianNLLLoss()
    
    validate = True

    # Instantiate model
    alphaDropout = True
    model = Net(hyperparams.hidden_size, alphaDropout, inputNumber, outputNumberState)
    
    # Optimizer
    opt = AdaBelief(model.parameters(), lr=hyperparams.lr, eps=1e-16, betas=(0.9,0.999), weight_decouple=True, rectify=False, print_change_log=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=hyperparams.lr_reduction_factor, patience=hyperparams.patience, verbose=True)

    # Initialize Net/Problem Configuration
    net_config = NetConfig(comm=comm, model=model, scheduler=scheduler, optimizer=opt, loss_fn_state=loss_fn_state, loss_fn_rew=loss_fn_rew, hyperparams=hyperparams, ifVal=validate, ifTest=False, dt_string=dt_string, launch=launch, scaleInputAndOutput=scaleIO, alphaDropout=alphaDropout, stateVariableCount, actionVariableCount)
    
    models={}
    for r in range(1, net_config.num_procs):
        models[r] = Net(net_config.hyperparams.hidden_size, alphaDropout, inputNumber, outputNumberState)
    #if rank == 0:
    #    print(models)
    
    argsEnv = ArgsEnv(dataPoints=0, totalDataPoints=0, previousDataPoints=0, previousTotalDataPoints=0, iterationSurrogate=0, scalers=None, rangeOutputState=None, varOutputState=None, rangeOutputRew=None, varOutputRew=None, dumpBestTrajectory=args.dumpBestTrajectory, initialRetrainingDataPoints=args.iniRetrain, retrainingDataPoints=args.retrain, keep_retraining=True, policyTestingEpisodes=policyTestingEpisodes, mode="Training", samplesTrained=[], models=models, surrUsage=[])
    
    #if rank == 0:
    #    print(f"[Ensemble] Optimizer = {opt}")
    #    print(f"[Ensemble] Validation = {validate}")
    #    print(f"[Ensemble] Testing = False")

    if rank == 0:
        fname = dirfiles["Dataset Train Korali"]
        if os.path.exists(fname):
            os.remove(fname)
        info = "#traj, step, s0, s1, ..., sn-1, a0, a1, ..., am-1, y0, y1, ..., yk-1, r, done"
        with open(fname, "w") as fhandle:
            fhandle.write(info + "\n")
            
        fname = dirfiles["Previous Dataset Train Korali"]
        if os.path.exists(fname):
            os.remove(fname)
        info = "#traj, step, s0, s1, ..., sn-1, a0, a1, ..., am-1, y0, y1, ..., yk-1, r, done"
        with open(fname, "w") as fhandle:
            fhandle.write(info + "\n")
            
        fname = dirfiles["Total Dataset Train"]
        if os.path.exists(fname):
            os.remove(fname)
        info = "#traj, step, s0, s1, ..., sn-1, a0, a1, ..., am-1, y0, y1, ..., yk-1, r, done"
        with open(fname, "w") as fhandle:
            fhandle.write(info + "\n")
            
        fname = dirfiles["Model Usage"]
        if os.path.exists(fname):
            os.remove(fname)
        info = ""
        with open(fname, "w") as fhandle:
            fhandle.write(info)
            
    elif rank > 0: # train surrogate model with initial dataset - ranks for training surrogate model
        # Other ranks still don't train
        #pass
        PATH_curr = dirfiles[f"Current Model {rank}"]
        PATH_prev = dirfiles[f"Previous Model {rank}"]
        PATH_check = dirfiles[f"Checkpoint Model {rank}"]
        if os.path.exists(PATH_curr):
            os.remove(PATH_curr)
        if os.path.exists(PATH_prev):
            os.remove(PATH_prev)
        if os.path.exists(PATH_check):
            os.remove(PATH_check)
            

    ####### Defining Korali Problem

    k = korali.Engine()
    e = korali.Experiment()

    ### Defining results folder and loading previous results, if any

    e.loadState(dirfiles["Results"] + '/latest');

    ### Initializing openAI Gym environment

    initEnvironment(e, envName, env, stateVariableCount, actionVariableCount, stateVariablesIndexes, net_config, tags, argsEnv, dirfiles, args)

    ### Defining Agent Configuration

    e["Solver"]["Type"] = "Agent / Continuous / VRACER"
    e["Solver"]["Mode"] = "Training"
    e["Solver"]["Episodes Per Generation"] = episodesPerGeneration
    e["Solver"]["Experiences Between Policy Updates"] = experiencesBetweenPolicyUpdates
    e["Solver"]["Learning Rate"] = args.lr
    e["Solver"]["Discount Factor"] = 0.995
    e["Solver"]["Mini Batch"]["Size"] = 256

    ### Setting Experience Replay and REFER settings

    e["Solver"]["Experience Replay"]["Start Size"] = startSize #131072
    e["Solver"]["Experience Replay"]["Maximum Size"] = maxSize #262144
    e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
    e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
    e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
    e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = args.opt

    e["Solver"]["Policy"]["Distribution"] = args.dis
    e["Solver"]["State Rescaling"]["Enabled"] = True
    e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
      
    ### Configuring the neural network and its hidden layers

    e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
    e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
    e["Solver"]["L2 Regularization"]["Enabled"] = args.l2 > 0.
    e["Solver"]["L2 Regularization"]["Importance"] = args.l2

    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

    ### Setting file output configuration

    e["Solver"]["Termination Criteria"]["Max Experiences"] = 10e6
    e["Solver"]["Experience Replay"]["Serialize"] = True
    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = 200
    e["File Output"]["Path"] = resultFolder
    
    if maxPolicyUpdates != 0:
        e["Solver"]["Termination Criteria"]["Max Policy Updates"] = maxPolicyUpdates

    ### Running Experiment

    if rank == 0:
    
        previousDataUsage = 0
    
        k.run(e)
        
        finalGenerationRes = e["Current Generation"]
        
        policyUpdateCount = e["Solver"]["Policy Update Count"]
        
        realRewardHistoryGenNum = []
        realRewardHistoryAverage = []
        realRewardHistoryStdev = []
        realRewardHistoryBest = []
        realRewardHistoryWorst = []
        for genNum in range(testingFrequency,finalGenerationRes+1): #,testingFrequency):
            fresname = dirfiles["Results"] + f"gen{genNum:08d}.json"
            if os.path.exists(fresname):
                with open(fresname) as fgen:
                    dataGen = json.load(fgen)
                    realRewardHistoryGenNum.append(genNum)
                    realRewardHistoryAverage.append(dataGen["Solver"]["Testing"]["Average Reward"])
                    realRewardHistoryStdev.append(dataGen["Solver"]["Testing"]["Stdev Reward"])
                    realRewardHistoryBest.append(dataGen["Solver"]["Testing"]["Best Reward"])
                    realRewardHistoryWorst.append(dataGen["Solver"]["Testing"]["Worst Reward"])
        realRewardHistory = np.transpose(np.array([realRewardHistoryGenNum, realRewardHistoryAverage, realRewardHistoryStdev, realRewardHistoryBest, realRewardHistoryWorst]))
        info_rew = "Generation,Average,Stdev,Best,Worst"
        np.savetxt(dirfiles["Real Testing Rewards"], realRewardHistory, delimiter=",", header=info_rew)#, fmt='%.5f')
        
        ignoreText = ""
        for r in range(1, num_procs):
            ignoreText += comm.recv(source=r, tag=tags["tag_retrained_ready"])
        
        argsEnv.keep_retraining = False
        
        # Need to take care of communication so all ranks return
        for r in range(1, net_config.num_procs):
            comm.send(argsEnv.keep_retraining, dest=r, tag=tags["tag_keep_retraining"])
            comm.send(argsEnv.iterationSurrogate, dest=r, tag=tags["tag_iter_surr"])
        
        print("[Korali] Target reached !!! (Stopping execution)")
        print(f"[Korali|Ensemble] Data Usage: {previousDataUsage}")
        fname_usage = dirfiles["Model Usage"]
        usageModelArray = np.genfromtxt(fname_usage)
        usagesurrogateModel = 100 * np.sum(usageModelArray)/np.size(usageModelArray) # 1's over total
        print(f"[Korali|Ensemble] Model Usage Total: {usagesurrogateModel:.4f}%")
        print("[Korali] Now, test trained ensemble on the total training dataset:")
        
        interactionsRealWarmUp = startSize #args.iniRetrain + args.retrain
        numberInteractionsWithReal = argsEnv.interactionsWithReal - interactionsRealWarmUp
        numberInteractionsWithSurr = argsEnv.interactionsWithSurr
        totalNumberInteractionsWithEnv = argsEnv.totalInteractionsWithEnv
        ratioExperiences = numberInteractionsWithReal / (numberInteractionsWithReal + numberInteractionsWithSurr)
        print(f"[Korali|Ensemble] Number Interactions With Real: {numberInteractionsWithReal}")
        print(f"[Korali|Ensemble] Number Interactions With Surr: {numberInteractionsWithSurr}")
        print(f"[Korali|Ensemble] Total Number Interactions With Env: {totalNumberInteractionsWithEnv}")
        print(f"[Korali|Ensemble] Warm Up Interactions: {interactionsRealWarmUp}")
        print(f"[Korali|Ensemble] Ratio Experiences: {ratioExperiences}")
        print(f"[Korali|Ensemble] Policy Update Count: {policyUpdateCount}")

        
        listInteractionsWithReal = argsEnv.listInteractionsWithReal
        listInteractionsWithSurr = argsEnv.listInteractionsWithSurr
        listSampleIds = argsEnv.listSampleIds
        interactionsHistory = np.transpose(np.array([listSampleIds, listInteractionsWithReal, listInteractionsWithSurr]))
        info_inter = "SampleId,InteractionsReal,InteractionsSurr"
        np.savetxt(dirfiles["Interactions"], interactionsHistory, delimiter=",", header=info_inter)#, fmt='%.5f')

        ### Testing policy on real environment, env2, dumping results
        
        print("\n-------------------------------------------------------------")
        print("Testing ...")
        print("-------------------------------------------------------------\n")
        
        #e["Problem"]["Environment Function"] = lambda s: env(s, net_config, tags, argsEnv) # real
        
        argsEnv.mode = "TestingReal"
        
        e["Solver"]["Mode"] = "Testing"
        e["Solver"]["Testing"]["Sample Ids"] = list(range(100))

        k.run(e)
        
        bestTestingEpisodeId = e["Solver"]["Testing"]["Best Episode Id"]
        averageTrainingReward = e["Solver"]["Training"]["Average Reward"]
        rewardHistory = e["Solver"]["Training"]["Reward History"]
        
        inputNumber = stateVariableCount + actionVariableCount
        outputNumberState = stateVariableCount
        outputNumberRew = 1
        usecols_X_train = tuple(range(2, inputNumber+2))
        usecols_y_train_state = tuple(range(inputNumber+2, inputNumber+2+outputNumberState))
        usecols_y_train_rew = (inputNumber+2+outputNumberState)
        total_data_X_train = np.genfromtxt(dirfiles["Total Dataset Train"], delimiter=",", skip_header=1, usecols=usecols_X_train)
        total_data_y_train_state = np.genfromtxt(dirfiles["Total Dataset Train"], delimiter=",", skip_header=1, usecols=usecols_y_train_state)
        total_data_y_train_rew = np.genfromtxt(dirfiles["Total Dataset Train"], delimiter=",", skip_header=1, usecols=usecols_y_train_rew)
        previousDataUsage = total_data_X_train.shape[0]

        averageTestReward = np.average(e["Solver"]["Testing"]["Reward"])
        print("Average Reward: " + str(averageTestReward))
        #if (averageTestReward < 150):
        #    print("Cartpole example did not reach minimum testing average.")
            
        totalCountVar = 0
        onesCountVar = 0
        for idx in range(bestTestingEpisodeId):
            l = argsEnv.surrUsage[idx]
            totalCountVar += len(l)
            onesCountVar += sum(l)
        usageSurrBestTestingPolicy = 100 * onesCountVar / totalCountVar
        print(f"[Korali|Ensemble] Model Usage at Best Policy Testing: {usageSurrBestTestingPolicy:.4f}%")
        
        iterationsUsed = argsEnv.iterationSurrogate-1
            
        fname_results = dirfiles["Results Launches"]
        #infores = "#previousDataUsage,usagesurrogateModel,finalGenerationRes,averageTrainingReward,bestTestingEpisodeId"
        info_res = f"{bestTestingEpisodeId},{averageTrainingReward:.4f},{finalGenerationRes},{previousDataUsage},{usageSurrBestTestingPolicy:.4f},{averageTestReward:.4f},{numberInteractionsWithReal},{numberInteractionsWithSurr},{totalNumberInteractionsWithEnv},{interactionsRealWarmUp},{ratioExperiences:.4f},{policyUpdateCount},{iterationsUsed}"
        with open(fname_results, "a") as fhandle_results:
            fhandle_results.write(info_res+"\n")
        
    else:
        
        keep_retraining = True
        retraining_iter = 0
        
        while keep_retraining:
        
            if retraining_iter > 0:
            
                if rank == 1:
                    print("[Rank 1] Start training ensemble ...")
            
                # retrain
                datasetKorali = dirfiles["Previous Dataset Train Korali"]
                    
                textPrint = trainEnsemble(net_config=net_config, datasetKorali=datasetKorali, iteration=retraining_iter, dirfiles=dirfiles)
                
                PATH_curr = dirfiles[f"Current Model {rank}"]
                PATH_prev = dirfiles[f"Previous Model {rank}"]
                
                if rank == 1:
                    print("[Rank 1] Finished training ensemble!!")
                shutil.copy(PATH_curr,PATH_prev)
                    
                comm.send(textPrint, dest=0, tag=tags["tag_retrained_ready"])
            
            keep_retraining = comm.recv(source=0, tag=tags["tag_keep_retraining"])
            retraining_iter = comm.recv(source=0, tag=tags["tag_iter_surr"])
            net_config.retraining_iter = retraining_iter

