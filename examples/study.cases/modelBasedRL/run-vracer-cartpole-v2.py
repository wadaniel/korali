#!/usr/bin/env python3

import os
import sys
import shutil
import json
import warnings
import argparse
from datetime import datetime
from mpi4py import MPI

#sys.path.append('./_modelCoord')
sys.path.append('./_model_cartpole')
from env_v2 import *
from model import Net, Hyperparams, NetConfig, trainEnsemble, predictEnsembleSerial, GaussianNLLLoss, VisualizationNetResults

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adabelief_pytorch import AdaBelief
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Arguments Surrogate Model based RL
parser = argparse.ArgumentParser()
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
parser.add_argument("--testFreq", type=int, default=5, help="testingFrequency")
parser.add_argument("--polTestEp", type=int, default=30, help="policyTestingEpisodes")
parser.add_argument("--expBetPolUp", type=float, default=1, help="Experiences Between Policy Updates")
parser.add_argument("--maxPolUp", type=float, default=0, help="Maximum Policy Updates")
parser.add_argument("--maxSize", type=float, default=10000, help="Maximum size of memory for experience replay")
parser.add_argument("--launchNum", type=int, default=1, help="Number of times to run vracer.")
parser.add_argument("--iniRetrain", type=int, default=10000, help="Initial retraining sample data points.")
parser.add_argument("--retrain", type=int, default=10000, help="Retraining sample data points.")
parser.add_argument("--lrRL", type=float, default=1e-4, help="Learning Rate RL")
parser.add_argument("--l2Regul", type=float, default=1.0, help="L2 Regularization importance RL, or False if 0.")
args = parser.parse_args()

comm = MPI.COMM_WORLD   # define communicator for the solver for surrogate model parallel training and predicting (Korali in serial mode)
rank = comm.Get_rank()
print("Rank:",rank)
num_procs = comm.Get_size()
print("Num procs:",num_procs)
tags = {"tag_keep_retraining": 11, "tag_iter_surr": 12, "tag_retrained_ready": 13}

now = datetime.now()
if args.m == "":
    args.m = "Results/" + now.strftime("%Y%m%d%H%M%S") + "/"
else:
    args.m = "Results/" + args.m

if rank == 0:
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
    
    dirfiles["Interactions"] = args.m + dt_string + f"interactions.csv"
    dirfiles["Real Testing Rewards"] = args.m + dt_string + f"real_rewards.csv"
    dirfiles["Dataset Train Korali"] = args.m + dt_string + f"dataset_train_korali.csv"
    dirfiles["Previous Dataset Train Korali"] = args.m + dt_string + f"dataset_train_korali_prev.csv"
    dirfiles["Total Dataset Train"] = args.m + dt_string + f"dataset_train_total.csv"
    dirfiles["Model Usage"] = args.m + dt_string + f"model_usage.csv"
    dirfiles["Cartpole Results"] = args.m + "_" + dt_string + "cartpoleTrainingResults"
    dirfiles["Best Trajectory"] = args.m + "_" + dt_string + "best_trajectory.csv"
    dirfiles["Experience Replay"] = args.m + dt_string + "experience_replay.csv"
    for r in range(1, num_procs):
        dirfiles[f"Current Model {r}"] = args.m + "Models/" + dt_string + f"Rank{r}_of_{num_procs}_surnet_curr.pth"
        dirfiles[f"Previous Model {r}"] = args.m + "Models/" + dt_string + f"Rank{r}_of_{num_procs}_surnet_prev.pth"
        dirfiles[f"Checkpoint Model {r}"] = args.m + "Models/" + dt_string + f"Rank{r}_of_{num_procs}_surnet_checkpoint.pth"
        dirfiles[f"Visualization {r}"] = args.m + "Visualization/" + dt_string + f"Rank{r}_of_{num_procs}_"

    if rank == 0:
        print("\n\n")
        print(f"\n************** Cartpole_v0 Surrogate - Launch={launch}/{args.launchNum} - " + dt_string + " **************\n")

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

    # Loss functions
    loss_fn_state = GaussianNLLLoss()
    loss_fn_rew = GaussianNLLLoss()
    
    validate = True

    # Instantiate model
    alphaDropout = True
    model = Net(hyperparams.hidden_size, alphaDropout)
    
    # Optimizer
    opt = AdaBelief(model.parameters(), lr=hyperparams.lr, eps=1e-16, betas=(0.9,0.999), weight_decouple=True, rectify=False, print_change_log=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=hyperparams.lr_reduction_factor, patience=hyperparams.patience, verbose=True)

    # Initialize Net/Problem Configuration
    net_config = NetConfig(comm=comm, model=model, scheduler=scheduler, optimizer=opt, loss_fn_state=loss_fn_state, loss_fn_rew=loss_fn_rew, hyperparams=hyperparams, ifVal=validate, ifTest=False, dt_string=dt_string, launch=launch, scaleInputAndOutput=scaleIO, alphaDropout=alphaDropout)
    
    models={}
    for r in range(1, net_config.num_procs):
        models[r] = Net(net_config.hyperparams.hidden_size, alphaDropout)
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
        info = "#traj, step, s0, s1, s2, s3, a, y0, y1, y2, y3, r, done"
        with open(fname, "w") as fhandle:
            fhandle.write(info + "\n")
            
        fname = dirfiles["Previous Dataset Train Korali"]
        if os.path.exists(fname):
            os.remove(fname)
        info = "#traj, step, s0, s1, s2, s3, a, y0, y1, y2, y3, r, done"
        with open(fname, "w") as fhandle:
            fhandle.write(info + "\n")
            
        fname = dirfiles["Total Dataset Train"]
        if os.path.exists(fname):
            os.remove(fname)
        info = "#traj, step, s0, s1, s2, s3, a, y0, y1, y2, y3, r, done"
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
            
    ### Setting results dir
    
    trainingResultsPath = dirfiles["Cartpole Results"]

    ####### Defining Korali Problem

    #TODO: moved import outside launch loop
    #import korali
    k = korali.Engine()
    e = korali.Experiment()

    ### Defining the Cartpole problem's configuration

    e["Problem"]["Type"] = "Reinforcement Learning / Continuous" # problem were we want to solve a sequential decision making problem in a continuous action domain
    e["Problem"]["Environment Function"] = lambda s: env(s, net_config, tags, argsEnv, dirfiles) # Computational Model - env
    e["Problem"]["Training Reward Threshold"] = trainingRewardThreshold #was 400 # Minimum value (r) of the episode’s average training reward for a policy to be considered as candidate.
    e["Problem"]["Testing Frequency"] = testingFrequency; # instead of e["Problem"]["Training Reward Threshold"] !!
    e["Problem"]["Policy Testing Episodes"] = policyTestingEpisodes #was 20 # Number of test episodes to run the policy (without noise) for, for which the average reward will serve to evaluate the reward termination criteria.
    e["Problem"]["Actions Between Policy Updates"] = 5 # Number of actions to take before requesting a new policy.

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
    e["Variables"][4]["Initial Exploration Noise"] = 1.0 # Initial standard deviation of the fixed exploration noise for the given action

    ### Defining Agent Configuration

    e["Solver"]["Type"] = "Agent / Continuous / VRACER"
    e["Solver"]["Mode"] = "Training" # Learns a policy for the reinforcement learning problem (other: "Testing" -> Tests the policy with a learned policy)
    e["Solver"]["Experiences Between Policy Updates"] = experiencesBetweenPolicyUpdates #was 10 # The number of experiences to receive before training/updating (real number, may be less than < 1.0, for more than one update per experience).
    e["Solver"]["Episodes Per Generation"] = episodesPerGeneration #was 1 # (= default) Indicates the how many finished episodes to receive in a generation (checkpoints are generated between generations).
    e["Solver"]["Concurrent Environments"] = 1

    # Off/On-policy RL
    # Note: Remember and Forget Experience Replay (ReF-ER), a novel method that can en- hance RL algorithms with parameterized policies (based on that)
    e["Solver"]["Experience Replay"]["Start Size"] = startSize #args.iniRetrain + args.retrain # The minimum number of experiences to gather before learning starts. - was 1000
    e["Solver"]["Experience Replay"]["Maximum Size"] = maxSize # The minimum number of experiences to accumulate before starting to forget.
    e#["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0 # (default)
    #e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1 # (default)
    #e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 0.0 # (default)
    #e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3 # (default)

    e["Solver"]["Discount Factor"] = 0.99 # Represents the weight given to the expectation of the cumulative reward from future experiences.
    e["Solver"]["Learning Rate"] = args.lrRL #1e-4 # The base learning rate to use for the NN hyperparameter optimization.
    e["Solver"]["Mini Batch"]["Size"] = 32 # The number of experiences to randomly select to train the neural network with (uniform distribution)

    if args.l2Regul != 0.:
        e["Solver"]["L2 Regularization"]["Enabled"] = True #False # (default)
        e["Solver"]["L2 Regularization"]["Importance"] = args.l2Regul #was 1.0 #0.0001 # (default)
        

    e["Solver"]["State Rescaling"]["Enabled"] = False # Determines whether to use state scaling (done only once after the initial exploration phase).
    e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False # Determines whether to use reward scaling
    e["Solver"]["Reward"]["Rescaling"]["Frequency"] = 1000 # The number of policy updates between consecutive reward rescalings.
    e["Solver"]["Policy"]["Distribution"] = "Normal" #was "Squashed Normal" # default - Use the a normal distribution for the production of an action given lower and upper bounds with compensation for normal gradients. Based on work by Guido Novati.

    ### Configuring the neural network and its hidden layers

    e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
    e["Solver"]["Neural Network"]["Optimizer"] = "AdaBelief"

    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

    ### Defining Termination Criteria

    #Note: episodes are number of environments fully executed, experiences are gathered in them, and generations are executions of the solver
    e["Solver"]["Termination Criteria"]["Max Generations"] = args.maxGen # Determines how many solver generations to run before stopping execution. Execution can be resumed at a later moment.
    e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = targetAverageReward #was 450 # The solver will stop when the given best average per-episode reward has been reached among the experiences between two learner updates.
    if maxPolicyUpdates != 0:
        e["Solver"]["Termination Criteria"]["Max Policy Updates"] = maxPolicyUpdates

    ### Setting file output configuration

    e["File Output"]["Path"] = trainingResultsPath
    e["File Output"]["Enabled"] = True #True
    e["File Output"]["Frequency"] = 1

    ### Console Verbosity:

    e["Console Output"]["Verbosity"] = "Detailed"

    ### Running Experiment

    if rank == 0:
    
        previousDataUsage = 0
    
        k.run(e)
        
        finalGenerationRes = e["Current Generation"]
        
        #policyUpdateCount = e["Solver"]["Policy Update Count"]
        
        realRewardHistoryGenNum = []
        realRewardHistoryAverage = []
        realRewardHistoryStdev = []
        realRewardHistoryBest = []
        realRewardHistoryWorst = []
        policyUpdateCounts = []
        for genNum in range(testingFrequency,finalGenerationRes+1,testingFrequency):
            with open(trainingResultsPath + f"/gen{genNum:08d}.json") as fgen:
                dataGen = json.load(fgen)
                realRewardHistoryGenNum.append(genNum)
                realRewardHistoryAverage.append(dataGen["Solver"]["Testing"]["Average Reward"])
                realRewardHistoryStdev.append(dataGen["Solver"]["Testing"]["Stdev Reward"])
                realRewardHistoryBest.append(dataGen["Solver"]["Testing"]["Best Reward"])
                realRewardHistoryWorst.append(dataGen["Solver"]["Testing"]["Worst Reward"])
                policyUpdateCounts.append(dataGen["Solver"]["Policy Update Count"])
        realRewardHistory = np.transpose(np.array([realRewardHistoryGenNum, realRewardHistoryAverage, realRewardHistoryStdev, realRewardHistoryBest, realRewardHistoryWorst, policyUpdateCounts]))
        info_rew = "Generation,Average,Stdev,Best,Worst,PolUpCount"
        np.savetxt(dirfiles["Real Testing Rewards"], realRewardHistory, delimiter=",", header=info_rew)#, fmt='%.5f')
        
        ignoreText = ""
        for r in range(1, num_procs):
            ignoreText += comm.recv(source=r, tag=tags["tag_retrained_ready"])
        
        argsEnv.keep_retraining = False
        
        # Need to take care of communication so all ranks return
        for r in range(1, net_config.num_procs):
            comm.send(argsEnv.keep_retraining, dest=r, tag=tags["tag_keep_retraining"])
            comm.send(argsEnv.iterationSurrogate, dest=r, tag=tags["tag_iter_surr"])
            
        bestTestingEpisodeId = e["Solver"]["Testing"]["Best Episode Id"]
        averageTrainingReward = e["Solver"]["Training"]["Average Reward"]
        rewardHistory = e["Solver"]["Training"]["Reward History"]
        
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
        policyUpdateCount = policyUpdateCounts[bestTestingEpisodeId-1]
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
        
        total_data_X_train = np.genfromtxt(dirfiles["Total Dataset Train"], delimiter=",", skip_header=1, usecols=(2,3,4,5,6))
        total_data_y_train_state = np.genfromtxt(dirfiles["Total Dataset Train"], delimiter=",", skip_header=1, usecols=(7,8,9,10))
        total_data_y_train_rew = np.genfromtxt(dirfiles["Total Dataset Train"], delimiter=",", skip_header=1, usecols=(11))
        previousDataUsage = total_data_X_train.shape[0]

        averageTestReward = np.average(e["Solver"]["Testing"]["Reward"])
        print("Average Reward: " + str(averageTestReward))
        if (averageTestReward < 150):
            print("Cartpole example did not reach minimum testing average.")
            
        totalCountVar = 0
        onesCountVar = 0
        for idx in range(bestTestingEpisodeId-argsEnv.discardedNum[bestTestingEpisodeId]):
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
