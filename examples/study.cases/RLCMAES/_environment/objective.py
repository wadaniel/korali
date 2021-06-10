#!/usr/bin/env python3

import numpy as np

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

class ObjectiveFactory:
  def __init__(self, populationSize):
    self.populationSize = populationSize
    self.mu = int(self.populationSize/2)
    self.weights = np.log(self.mu+1/2)-np.log(np.array(range(self.mu))+1)
    self.weights /= sum(self.weights)
    self.a = 11
    self.b = 7
    self.mean = np.zeros(2)
    self.scale = 0.1
    self.sigma = np.diag(np.ones(2))
    self.population = np.zeros( (2,self.populationSize ) )
    self.feval = np.zeros( self.populationSize )
    self.bestEver = np.inf
    self.prevBestF = np.inf
    self.curBestF = np.inf
    self.initialBestF = np.inf
    self.initialEf = np.inf
    self.prevEf = np.inf
    self.curEf = np.inf
    self.function = None
    self.step = 0

  def reset(self, noise=True):
    self.a = self.a 
    self.b = self.b 
    if noise:
        self.a += np.random.uniform(-1., 1.)
        self.b += np.random.uniform(-1., 1.)

    if np.random.uniform(0.,1.) > 0.5:
        self.function = lambda x : (x[0]**2+x[1]-self.a)**2 + (x[0]+x[1]**2-self.b)**2 # Himmelblau
    else:
        self.function = lambda x : (x[0]+2.*x[1]-self.a)**2 + (2*x[0]+x[1]-self.b)**2 # Booth
    #self.function = lambda x : self.a*(x[1]-x[0]**2)**2 + (self.b - x[0])**2 # Rosenbrock
    #self.function = lambda x : np.sin(x[0]+x[1])+(x[0]-x[1]**2)-self.a*x[0]+self.b*x[1]+1 # McCormick
    self.population = np.random.multivariate_normal(self.mean, self.scale*self.sigma, self.populationSize)
    self.evalPopulation()
    self.initialBestF = self.curBestF
    self.initialEf = self.curEf
    self.step = 0

  def isOver(self):
    return False

  def evalPopulation(self):
    
    # Evaluate and sort
    self.feval = np.array([ self.function(self.population[i]) for i in range(self.populationSize) ])
    sortIdx = np.argsort(self.feval)
    self.feval = self.feval[sortIdx]
    self.population = np.array(self.population[sortIdx])

    # Update previous 
    self.prevEf = self.curEf
    self.prevBestF = self.curBestF
    
    # Update current best values
    self.curEf = np.mean(self.feval)
    self.curBestF = min(self.feval)
    if self.curBestF < self.bestEver:
        self.bestEver = self.curBestF

  def advance(self, action):
    cm = np.clip(action[0], a_min=0.0, a_max=1.0)
    cs = action[1]
    meanOfBest = np.average(self.population[:self.mu], axis=0, weights=self.weights)
    self.mean = (1.-cm) * self.mean + cm * meanOfBest
    self.scale *= np.exp(cs/2)
    self.population = np.random.multivariate_normal(self.mean, self.scale*self.sigma, self.populationSize)
    self.evalPopulation()
    self.step += 1

  def getState(self):
    state = np.zeros(3*self.populationSize)
    for i in range(self.populationSize):
        state[i*3:i*3+2] = self.population[i]
        state[i*3+2] = self.feval[i]/self.initialEf
    return state

  def getReward(self):

    return (self.prevEf - self.curEf)/self.initialEf

if __name__ == '__main__':
    objective = RandomHimmelblau(3)
    objective.reset()
    objective.advance(np.ones(2))
    state = objective.getState()
    reward = objective.getReward()
