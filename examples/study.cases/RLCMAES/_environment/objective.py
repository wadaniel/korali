#!/usr/bin/env python3

import numpy as np

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

class RandomHimmelblau:
  def __init__(self, populationSize):
    self.populationSize = populationSize
    self.a = -11
    self.b = -7
    self.mu = np.zeros(2)
    self.sigma = 0.2*np.diag(np.ones(2))
    self.population = np.zeros( (2,self.populationSize ) )
    self.feval = np.zeros( self.populationSize )
    self.bestEver = np.inf
    self.prevBestF = np.inf
    self.curBestF = np.inf
    self.initalBestF = np.inf
    self.function = None

  def reset(self):
    self.a = self.a + np.random.uniform(-1, 1)
    self.b = self.b + np.random.uniform(-1, 1)
    self.function = lambda x : (x[0]**2+x[1]-self.a)**2 + (x[0]+x[1]**2-self.b)**2
    self.population = np.random.multivariate_normal(self.mu, self.sigma, self.populationSize)
    self.evalPopulation()
    self.initalBestF = self.curBestF
    self.step = 0

  def isOver(self):
    return False

  def evalPopulation(self):
    for i in range(self.populationSize):
        self.feval[i] = self.function(self.population[i])
    
    self.prevBestF = self.curBestF
    self.curBestF = min(self.feval)
    if self.curBestF < self.bestEver:
        self.bestEver = self.curBestF

  def advance(self, action):
    self.mu += action
    self.population = np.random.multivariate_normal(self.mu, self.sigma, self.populationSize)
    self.evalPopulation()

  def getState(self):
    state = np.zeros(3*self.populationSize)
    for i in range(self.populationSize):
        state[i*3:i*3+2] = self.population[i]
        state[i*3+2] = self.feval[i]
    return state

  def getReward(self):
    return 0.001 * (self.prevBestF - self.curBestF)

if __name__ == '__main__':
    objective = RandomHimmelblau(3)
    objective.reset()
    objective.advance(np.ones(2))
    state = objective.getState()
    reward = objective.getReward()
