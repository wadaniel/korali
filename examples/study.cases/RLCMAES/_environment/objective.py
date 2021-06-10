#!/usr/bin/env python3

import numpy as np

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

class ObjectiveFactory:
  def __init__(self, populationSize):
    self.populationSize = populationSize
    self.a = -11
    self.b = -7
    self.mu = np.zeros(2)
    self.scale = 1.0
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

  def reset(self):
    self.a = self.a #+ np.random.uniform(-0.1, 0.1)
    self.b = self.b #+ np.random.uniform(-0.1, 0.1)
    self.function = lambda x : (x[0]**2+x[1]-self.a)**2 + (x[0]+x[1]**2-self.b)**2 # Himmelblau
    #self.function = lambda x : np.sin(x[0]+x[1])+(x[0]-x[1]**2)-self.a*x[0]+self.b*x[1]+1 # McCormick
    #self.function = lambda x : (x[0]+2.*x[1]-self.a)**2 + (2*x[0]+x[1]-self.b)**2 # Booth
    #self.function = lambda x : self.a*(x[1]-x[0]**2)**2 + (self.b - x[0])**2 # Rosenbrock
    self.population = np.random.multivariate_normal(self.mu, self.scale*self.sigma, self.populationSize)
    self.evalPopulation()
    self.initialBestF = self.curBestF
    self.initialEf = self.curEf
    self.step = 0

  def isOver(self):
    return False

  def evalPopulation(self):
    
    self.feval = np.array([ self.function(self.population[i]) for i in range(self.populationSize) ])
    sortIdx = np.argsort(self.feval)
    self.feval = self.feval[sortIdx]
    self.population = self.population[sortIdx]

    #print(self.feval)
    #print(self.population)

    self.prevEf = self.curEf
    self.curEf = np.mean(self.feval)
    self.prevBestF = self.curBestF
    self.curBestF = min(self.feval)
    if self.curBestF < self.bestEver:
        self.bestEver = self.curBestF

  def advance(self, action):
    self.mu = action[:2]
    self.scale *= np.exp(action[2])
    self.population = np.random.multivariate_normal(self.mu, self.scale*self.sigma, self.populationSize)
    self.evalPopulation()

  def getState(self):
    state = np.zeros(2*self.populationSize)
    for i in range(self.populationSize):
        state[i*2:i*2+2] = self.population[i]
        #state[i*3+2] = self.feval[i]
    return state

  def getReward(self):

    return 0.001*(self.prevEf - self.curEf)

if __name__ == '__main__':
    objective = RandomHimmelblau(3)
    objective.reset()
    objective.advance(np.ones(2))
    state = objective.getState()
    reward = objective.getReward()
