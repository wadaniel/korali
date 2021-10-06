#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
from mnist import MNIST

k = korali.Engine()
e = korali.Experiment()

### Hyperparameters
 
learningRate = 0.0001
decay = 0.0001
trainingBatchSize = 60
epochs = 90

### Getting MNIST data

mndata = MNIST('./_data')
mndata.gz = True
trainingImages, trainingLabels = mndata.load_training()
testingImages, testingLabels = mndata.load_testing()

### Converting images to Korali form (requires a time dimension)

trainingImageVector = [ [ x ] for x in trainingImages ]
testingImageVector = [ [ x ] for x in testingImages ]

### Converting label data to (0,1) vector form

trainingLabelVector = [ ]
for l in trainingLabels:
 newtrainingLabel = np.zeros(10).tolist()
 newtrainingLabel[l] = 1
 trainingLabelVector.append(newtrainingLabel)

testingLabelVector = [ ]
for l in testingLabels:
 newLabel = np.zeros(10).tolist()
 newLabel[l] = 1
 testingLabelVector.append(newLabel)

### Shuffling training data set for stochastic gradient descent training

jointSet = list(zip(trainingImageVector, trainingLabelVector))
random.shuffle(jointSet)
trainingImageVector, trainingLabelVector = zip(*jointSet)
 
### Calculating Derived Values

stepsPerEpoch = int(len(trainingImageVector) / trainingBatchSize)
testingBatchSize = len(testingLabelVector)
 
### If this is test mode, run only one epoch
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  epochs=1
  stepsPerEpoch=1

### Configuring general problem settings

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = trainingBatchSize
e["Problem"]["Inference Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = 10

### Using a neural network solver (deep learning) for inference

e["Solver"]["Termination Criteria"]["Max Generations"] = 1
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Steps Per Generation"] = 1

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

## Convolutional Layer with tanh activation function [1x28x28] -> [6x28x28]
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"]      = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"]       = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"]     = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"]       = 2
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"]    = 2
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"]   = 6*28*28

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

## Learnable Pooling Layer [6x28x28] -> [6x14x14]
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Height"]      = 28
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Width"]       = 28
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"]   = 6*14*14

## Convolutional Layer with tanh activation function [6x14x14] -> [16x10x10]
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"]      = 14
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"]       = 14
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"]     = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"]   = 16*10*10

e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/Tanh"

## Learnable Pooling Layer [16x10x10] -> [16x5x5]
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Height"]      = 10
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Width"]       = 10
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Output Channels"]   = 16*5*5

## Fully Connected Convolutional Layer with tanh activation function [16x5x5] -> [120x1x1]
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"]       = 5
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"]     = 5
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"]   = 120

e["Solver"]["Neural Network"]["Hidden Layers"][7]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Function"] = "Elementwise/Tanh"

## Fully Connected Layer with tanh
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Output Channels"] = 84

e["Solver"]["Neural Network"]["Hidden Layers"][9]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Function"] = "Elementwise/Tanh"

## Fully Connected Layer with softmax
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Output Channels"] = 10

e["Solver"]["Neural Network"]["Hidden Layers"][11]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Function"] = "Softmax"

### Configuring output

e["Console Output"]["Verbosity"] = "Silent"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Printing Configuration

print("[Korali] Running MNIST solver.")
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImageVector)))
print("[Korali] Batch Size: " + str(trainingBatchSize))
print("[Korali] Epochs: " + str(epochs))
print("[Korali] Initial Learning Rate: " + str(learningRate))
print("[Korali] Decay: " + str(decay))

### Running SGD loop

for epoch in range(epochs):
 for step in range(stepsPerEpoch):
 
  # Creating minibatch
  miniBatchInput = trainingImageVector[step * trainingBatchSize : (step+1) * trainingBatchSize]
  miniBatchSolution = trainingLabelVector[step * trainingBatchSize : (step+1) * trainingBatchSize]
  
  # Passing minibatch to Korali
  e["Problem"]["Input"]["Data"] = miniBatchInput
  e["Problem"]["Solution"]["Data"] = miniBatchSolution
 
  # Reconfiguring solver
  e["Solver"]["Learning Rate"] = learningRate
  e["Solver"]["Termination Criteria"]["Max Generations"] = e["Solver"]["Termination Criteria"]["Max Generations"] + 1
  
  # Running step
  k.run(e)
  
 # Printing Information
 print("[Korali] --------------------------------------------------")
 print("[Korali] Epoch: " + str(epoch) + "/" + str(epochs))
 print("[Korali] Learning Rate: " + str(learningRate))
 print('[Korali] Current Training Loss: ' + str(e["Solver"]["Current Loss"])) 
    
 # Evaluating testing set
 testingInferredVector = testInferredSet = e.getEvaluation(testingImageVector)
 
 # Getting loss for testing set
 squaredMeanError = 0.0
 for i, res in enumerate(testingInferredVector):
  sol = testingLabelVector[i] 
  for j, s in enumerate(sol):
   diff = res[j] - sol[j]
   squaredMeanError += diff * diff 
 squaredMeanError = squaredMeanError / (float(testingBatchSize) * 2.0)
 print('[Korali] Current Testing Loss:  ' + str(squaredMeanError))
 
 # Adjusting learning rate via decay
 learningRate = learningRate * (1.0 / (1.0 + decay * (epoch+1)));
 
