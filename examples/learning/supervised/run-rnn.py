#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import korali
k = korali.Engine()

# Setting random seed for reproducibility
np.random.seed(0xC0FFEE)

# Input parameters
Arch = "FNN"
#Arch = "RNN"
tf = 2.0 # Total Time
dt = 0.5 # Time Differential
B = 100 # Batch Size
s = 1.0  # Parameter for peak separation
w = np.pi # Parameter for wave speed
a = 1.0 # Scaling

# Transformation Function
def y(x, t): return np.sin(x * s +  w * t)  

X = np.random.uniform(0, np.pi*2, B)
T = np.arange(0, tf, dt)

trainingInputSet = [ ]
for i, t in enumerate(T):
 trainingInputSet.append([ ])
 for j, x in enumerate(X):
  trainingInputSet[i].append([x,t])
   
trainingSolutionSet = [ ] 
for i, t in enumerate(T):
 trainingSolutionSet.append([ ])
 for j, x in enumerate(X):
  trainingSolutionSet[i].append([y(x, t)])

### Defining a learning problem to infer values of sin(x)

e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Inputs"] = trainingInputSet
e["Problem"]["Solution"] = trainingSolutionSet

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Steps Per Generation"] = 10
e["Solver"]["Optimizer"] = "AdaBelief"
e["Solver"]["Learning Rate"] = 0.0001

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "CuDNN"

if (Arch == "FNN"):

 e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Input"
 e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 2
 
 e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Layer/FeedForward"
 e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 64
 
 e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Activation"
 e["Solver"]["Neural Network"]["Layers"][2]["Function"] = "Elementwise/Tanh"
 
 e["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Layer/FeedForward"
 e["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 64
 
 e["Solver"]["Neural Network"]["Layers"][4]["Type"] = "Layer/Activation"
 e["Solver"]["Neural Network"]["Layers"][4]["Function"] = "Elementwise/Tanh"
 
 e["Solver"]["Neural Network"]["Layers"][5]["Type"] = "Layer/FeedForward"
 e["Solver"]["Neural Network"]["Layers"][5]["Node Count"] = 1
 
 e["Solver"]["Neural Network"]["Layers"][6]["Type"] = "Layer/Output"
 
else:

 e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Input"
 e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 2
 
 e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Layer/FeedForward"
 e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 64
 
 e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Recurrent"
 e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 64
 e["Solver"]["Neural Network"]["Layers"][2]["Mode"] = "GRU"
 
 e["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Layer/FeedForward"
 e["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 1
 
 e["Solver"]["Neural Network"]["Layers"][4]["Type"] = "Layer/Output"

e["Console Output"]["Frequency"] = 1
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

e["Solver"]["Termination Criteria"]["Max Generations"] = 100
k.run(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

testInputSet = [ ]
for i, t in enumerate(T):
 testInputSet.append([ ])
 for j, x in enumerate(X):
  testInputSet[i].append([x,t])
   
testSolutionSet = [ ] 
for i, t in enumerate(T):
 testSolutionSet.append([ ])
 for j, x in enumerate(X):
  testSolutionSet[i].append([y(x, t)])

testInferredSet = e.getEvaluation(testInputSet) 

cmap = cm.get_cmap(name='Set1')

for i, t in enumerate(T):
 xAxis = [ x[0] for x in testInputSet[i] ]
 plt.plot(xAxis, testSolutionSet[i], "o", color=cmap(i))
 plt.plot(xAxis, testInferredSet[i], "x", color=cmap(i))

### Calc MSE on test set

#mse = np.mean((np.array(testInferredSet) - np.array(testOutputSet))**2)
#print("MSE on test set: {}".format(mse))

### Plotting Results

plt.show()
