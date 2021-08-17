#!/usr/bin/env python3

import numpy as np

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

class ObjectiveFactory:
  def __init__(self, populationSize):

    # Initialize constants
    self.populationSize = populationSize
    self.mu = int(self.populationSize/2)
    self.weights = np.log(self.mu+1/2)-np.log(np.array(range(self.mu))+1)
    self.ueff = sum(self.weights)**2/sum(self.weights**2)
    self.weights /= sum(self.weights)
    self.dim = 2

    self.chi = np.sqrt(self.dim)*(1-1/(4*self.dim)+1/(21*self.dim**2))
    self.dhat = 1. + 2. * max(0, np.sqrt((self.ueff-1)/(self.dim+1))-1.) # d - cp
    self.cc = (4.+self.ueff/self.dim)/(self.dim+4.+2.*self.ueff/self.dim)
    self.c1 = 2./((self.dim+1.3)**2+self.ueff)
    self.cu = min(1.-self.c1, 2.*(self.ueff-2+1/self.ueff)/((self.dim+2)**2+2.*self.ueff/2))
   
    self.cs = (self.ueff+2.)/(self.dim+self.ueff+5)
    self.cm = 1.

    # Init variables
    self.reset()

  def reset(self, noise=1.0):

    # Initialize variable params
    self.scale = self.cs
    self.mean = np.zeros(self.dim)
    self.cov = np.diag(np.ones(self.dim))
    self.paths = np.zeros(self.dim)
    self.pathc = np.zeros(self.dim)

    self.population = np.zeros( (self.dim,self.populationSize ) )
    self.feval = np.zeros( self.populationSize )
    self.prevBestEver = np.inf
    self.bestEver = np.inf
    self.prevBestF = np.inf
    self.curBestF = np.inf
    self.initialBestF = np.inf
    self.initialEf = np.inf
    self.prevEf = np.inf
    self.curEf = np.inf
    self.name = "NA"
    self.function = None
    self.step = 0

    # Initialize optimziation target
    #u = 0.0
    u = np.random.uniform(0.,1./5) + 2./5 # Rosenrbock
    self.a = 0.
    self.b = 0.
    self.noise = noise
    self.a += self.noise*np.random.uniform(-1., 1.)
    self.b += self.noise*np.random.uniform(-1., 1.)

    # Choose function to optimize
    if u < 1./5:
        self.name = "Himmelblau"
        self.function = lambda x : (x[0]**2+x[1]-self.a)**2 + (x[0]+x[1]**2-self.b)**2 # Himmelblau
    elif u < 2./5:
        self.name = "Booth"
        self.function = lambda x : (x[0]+2.*x[1]-self.a)**2 + (2*x[0]+x[1]-self.b)**2 # Booth
    elif u < 3./5:
        self.name = "Rosenbrock"
        self.function = lambda x : abs(self.a)*(x[1]-x[0]**2)**2 + (self.b - x[0])**2 # Rosenbrock
    elif u < 4./5:
        self.name = "Spheres"
        self.function = lambda x : (self.a-x[0])**2 + (self.b-x[1])**2 # Two Spheres
    else:
        self.name = "Levi"
        self.function = lambda x : np.sin(3.*np.pi*(x[0]-self.a))**2 + (x[0]-self.a-1.)**2*(1+np.sin(3*np.pi*(x[1]-self.b))**2)+(x[1]-self.b-1.)**2*(1.+np.sin(2.*np.pi*(x[1]-self.b))**2) # Levi
    
    #self.function = lambda x : np.sin(x[0]+x[1])+(x[0]-x[1]**2)-self.a*x[0]+self.b*x[1]+1 # McCormick

    # Initialize first population
    self.population = np.random.multivariate_normal(self.mean, self.scale*self.scale*self.cov, self.populationSize)

    # Evaluate
    self.evalPopulation()

    # Store initial evals
    self.prevBestEver = self.curBestF
    self.initialBestF = self.curBestF
    self.initialEf = self.curEf

    # Init step
    self.step = 0

  def evaluateNegative(self, s):
      x = s["Parameters"]
      res = -self.function(x)
      s["F(x)"] = res

  def isOver(self):
    return False

  def evalPopulation(self):
    
    # Evaluate and sort
    self.feval = np.array([ self.function(self.population[i]) for i in range(self.populationSize) ]) + 1e-32
    sortIdx = np.argsort(self.feval)
    self.feval = self.feval[sortIdx]
    self.population = np.array(self.population[sortIdx])

    # Update previous 
    self.prevEf = self.curEf
    self.prevBestF = self.curBestF
    
    # Update current best values
    self.curEf = np.mean(self.feval)
    self.curBestF = min(self.feval)
    assert self.curBestF > 0., "Best must be positive {} ({})".format(self.curBestF, self.name)
    if self.curBestF < self.bestEver:
        self.bestEver = self.curBestF

  def advance(self, action):
    cs = np.clip(action[0], a_min=0.0, a_max=1.0)
    #cm = np.clip(action[1], a_min=0.0, a_max=1.0)
    #cs = self.cs
    cm = self.cm
    y = (self.population[:self.mu]-self.mean)/self.scale

    # Calc weighted mean and cov
    weightedMeanOfBest = np.average(y, weights=self.weights, axis=0)

    covOfBest = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        covOfBest += self.weights[i] * np.outer(y[i], y[i])

    # Update mean
    prevMean = self.mean
    self.mean += cm * self.scale * weightedMeanOfBest

    # Update step size & path
    self.paths = (1-cs)*self.paths+np.sqrt(cs*(2.-cs)*self.ueff)*np.dot(np.linalg.inv(np.linalg.cholesky(self.cov)), weightedMeanOfBest)
    self.scale *= np.exp(cs/(self.dhat-cs)*(np.linalg.norm(self.paths)/self.chi-1))
    if(self.scale < 1e-24):
        print("Warning: scale reaching lower bound")
        self.scale = 1e-24

    # Update cov & path
    hsig = 0.
    if cs < 1e-6:
        hsig = 0.
    elif np.linalg.norm(self.paths)/np.sqrt(1-(1-cs)**(2.*self.step+1)) < 1.4+2/(self.dim+1)*self.chi:
        self.hsig = 1.
    dhsig = min((1.-hsig)*self.cc*(2.-self.cc),1.0)

    self.pathc = (1-self.cc)*self.pathc+np.sqrt(self.cc*(2.-self.cc)*self.ueff)*weightedMeanOfBest
    self.cov = (1.+self.c1*dhsig-self.c1-self.cu) * self.cov + self.c1*np.outer(self.pathc, self.pathc) + self.cu * covOfBest

    # Resample
    self.population = np.random.multivariate_normal(self.mean, self.scale**2*self.cov, self.populationSize)

    # Evaluate
    self.evalPopulation()
    self.step += 1

  def getState(self):
    state = np.zeros((self.dim+1)*self.mu+1+self.dim)
    for i in range(self.mu):
        state[i*(self.dim+1):i*(self.dim+1)+self.dim] = self.population[i]
        state[i*(self.dim+1)+self.dim] = self.feval[i]/self.curEf
    state[-1-self.dim:-1] = self.scale**2*np.diag(self.cov) # diagonal variance
    state[-1] = self.bestEver/self.curEf # relative function eval
    #state[-1] = self.bestEver/self.curEf # relative function eval
    assert np.any(np.isfinite(state) == False) == False, "State not finite {}".format(state)
    return state

  def getReward(self):
    #r = (self.prevEf - self.curEf)/self.initialEf
    #r = (self.prevEf - self.curEf)/self.prevEf
    #r = +np.log(self.prevEf)-np.log(self.curEf)
    r = -np.log(self.curEf)
    assert np.isfinite(r), "Return not finite {}".format(r)

    return r

if __name__ == '__main__':
    objective = ObjectiveFactory(8)
    objective.reset()
    objective.advance(np.ones(3))
    state = objective.getState()
    reward = objective.getReward()
