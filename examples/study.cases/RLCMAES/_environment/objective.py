#!/usr/bin/env python3

import sys
import numpy as np

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

class ObjectiveFactory:
  def __init__(self, objective, dim, populationSize):

    # Initialize constants
    self.objective = objective
    self.populationSize = populationSize
    self.mu = int(self.populationSize/2)
    self.weights = np.log(self.mu+1/2)-np.log(np.array(range(self.mu))+1)
    self.ueff = sum(self.weights)**2/sum(self.weights**2)
    self.weights /= sum(self.weights)
    self.dim = dim

    self.chi = np.sqrt(self.dim)*(1-1/(4*self.dim)+1/(21*self.dim**2))
    self.dhat = 1. + 2. * max(0, np.sqrt((self.ueff-1)/(self.dim+1))-1.) # d - cp
    self.cc = (4.+self.ueff/self.dim)/(self.dim+4.+2.*self.ueff/self.dim)
    self.c1 = 2./((self.dim+1.3)**2+self.ueff)
    self.cu = min(1.-self.c1, 2.*(self.ueff-2+1/self.ueff)/((self.dim+2)**2+2.*self.ueff/2))
   
    self.cs = (self.ueff+2.)/(self.dim+self.ueff+5)
    self.cm = 1.

    # Init variables
    self.reset()

  def reset(self, noise=0.0):

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
    self.function = None
    self.step = 0

 
    # Init random objective if desired
    if self.objective == "random":
        nobjectives = 7
        u = np.random.uniform(0.,1.)

        if u < 1/nobjectives:
            self.objective = "fsphere"
 
        elif u < 2/nobjectives:
            self.objective = "felli"

        elif u < 3/nobjectives:
            self.objective = "fcigar"

        elif u < 4/nobjectives:
            self.objective = "ftablet"

        elif u < 5/nobjectives:
            self.objective = "fcigtab"

        elif u < 6/nobjectives:
            self.objective = "ftwoax"
 
        else:
            self.objective = "fdiffpow"
 
    # Init random noise params
    self.noise = noise
    self.a = self.noise*np.random.uniform(-1., 1.)
    self.b = self.noise*np.random.uniform(-1., 1.)
    self.zero = self.noise*np.random.uniform(-1., 1.)

    # Set function to optimize
    if self.objective == "himmelblau":
        self.a += -11
        self.b += -7
        self.function = lambda x : (np.sum(np.power(x[:int(self.dim/2)],2))+np.sum(x[int(self.dim/2):])+self.a)**2 + (np.sum(x[:int(self.dim/2):])+np.sum(np.power(x[int(self.dim/2):],2))+self.b)**2

    elif self.objective == "booth":
        # zero at (1,3)
        self.a += -7
        self.b += -5
        self.function = lambda x : (np.sum(x[:int(self.dim/2)])+np.sum(x[int(self.dim/2):])*2+self.a)**2 + (np.sum(x[:int(self.dim/2):])*2+np.sum(x[int(self.dim/2):])+self.b)**2

    elif self.objective == "rosenbrock":
        # zero at b
        self.a += 100
        self.b += 1 
        self.function = lambda x : abs(self.a)*np.sum((x[1:]-np.power(x[:-1],2))**2) + np.sum(np.power(np.subtract(x, self.b),2))

    elif self.objective == "spheres":
        # zero at b
        self.a += +5
        self.b += -5
        self.function = lambda x : abs(self.a)*np.sum(np.power(np.subtract(self.b,x),2)) 

    elif self.objective == "levi":
        # zero at a
        self.a += -1
        self.b += 10
        self.function = lambda x : np.sin(np.pi*(1+(x[0]+self.a)/4))**2 + \
                np.sum( ((x[:-1]+self.a)/4)**2 * (1+self.b*np.power(np.sin(np.pi*(1+(x[:-1]+self.a)/4)+1),2)) ) + \
                ((x[-1]+self.a)/4)**2 * (1+np.sin(2*np.pi*(1+(x[-1]+self.a)/4)))

    elif self.objective == "levi13":
        # zero at (1,1)
        self.a = -1
        self.b = 10
        self.function = lambda x : np.sin(3.*np.pi*(1+(x[0]+self.a)/4))**2 + \
                np.sum(((x[:-1]+self.a)/4)**2*(1+self.b*np.sin(np.pi*(1+(x[:-1]+self.a)/4)+1)**2)) + \
                ((x[-1]+self.a)/4)**2*(1+np.sin(2*np.pi*(1+(x[-1]+self.a)/4)))

    elif self.objective == "ackley":
        # zero at b
        self.a += 20
        self.b += 0
        self.function = lambda x : -self.a*np.exp(-0.2*np.sqrt(1/self.dim*np.sum(np.power(x-self.b, 2))))-np.exp(1/self.dim*np.sum(np.cos(2.*np.pi*(x-self.b)))) + self.a + np.exp(1)
 
    elif self.objective == "bukin":
        # zero at (0.1*b, 0.1*b, .., -b)
        self.a += 100
        self.b += 10
        self.function = lambda x : abs(self.a)*np.sum(np.sqrt(np.abs(x[1:]-0.01*x[-1]**2))) + 0.01*np.abs(x[-1]+self.b)
    
    elif self.objective == "fsphere":
        self.zero += 0
        self.function = lambda x : np.sum(np.power(x-self.zero,2))

    elif self.objective == "felli":
        self.zero += 0
        self.function = lambda x : np.sum(np.power(10,6*np.arange(self.dim)/self.dim)*np.power(x-self.zero,2))
 
    elif self.objective == "fcigar":
        self.zero += 0
        self.function = lambda x : (x[0]-self.zero)**2+np.sum(10**6*np.power(x[1:]-self.zero,2))
 
    elif self.objective == "ftablet":
        self.zero += 0
        self.function = lambda x : 10**6*(x[0]-self.zero)**2+np.sum(np.power(x[1:]-self.zero,2))

    elif self.objective == "fcigtab":
        self.zero += 0
        self.function = lambda x : (x[0]-self.zero)**2+np.sum(10**4*np.power(x[1:-1]-self.zero,2))+10**8*(x[-1]-self.zero)**2
 
    elif self.objective == "ftwoax":
        self.zero += 0
        self.function = lambda x : np.sum(10**6*np.power(x[:int(self.dim/2)]-self.zero,2))+np.sum((x[int(self.dim/2):]-self.zero)**2)
 
    elif self.objective == "fparabr":
        self.zero += 0
        self.function = lambda x : -x[0]+100*np.sum(np.power(x[1:]-self.zero,2))
 
    elif self.objective == "fsharpr":
        self.zero += 0
        self.function = lambda x : -x[0]+100*np.sqrt(np.sum(np.power(x[1:]-self.zero,2)))

    elif self.objective == "fdiffpow":
        self.zero += 0
        self.function = lambda x : np.sum(np.power(np.abs(x-self.zero),2+10*np.arange(self.dim)/self.dim))







    else:
        print("Objective {} not recognized! Abort..".format(self.objective))
        sys.exit()


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
    assert self.curBestF > 0., "Best must be positive {} ({})".format(self.curBestF, self.objective)
    if self.curBestF < self.bestEver:
        self.bestEver = self.curBestF

  def advance(self, action):

    # Unpack actions
    cs = np.clip(action[0], a_min=0.0, a_max=1.0)
    cm = np.clip(action[1], a_min=0.0, a_max=1.0)
    #csaddon = np.clip(action[2], a_min=-0.5, a_max=0.5)
    
    # Calc weighted mean and cov
    y = (self.population[:self.mu]-self.mean)/self.scale
    weightedMeanOfBest = np.average(y, weights=self.weights, axis=0)

    covOfBest = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        covOfBest += self.weights[i] * np.outer(y[i], y[i])

    # Update mean
    prevMean = self.mean
    self.mean += cm * self.scale * weightedMeanOfBest

    # Update step size & path
    self.paths = (1-cs)*self.paths+np.sqrt(cs*(2.-cs)*self.ueff)*np.dot(np.linalg.inv(np.linalg.cholesky(self.cov)), weightedMeanOfBest)
    self.scale *= np.exp(cs/self.dhat*(np.linalg.norm(self.paths)/self.chi-1))
    #self.scale *= np.exp(cs/self.dhat*(np.linalg.norm(self.paths)/self.chi-1) + csaddon)
    
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
    state = np.zeros((self.dim+1)*self.mu+1)
    diagsdev = self.scale*np.sqrt(np.diag(self.cov)) # diagonal sdev
    for i in range(self.mu):
        state[i*(self.dim+1):i*(self.dim+1)+self.dim] = np.divide(self.population[i] - self.mean, diagsdev)
        state[i*(self.dim+1)+self.dim] = self.feval[i]/self.curEf
    state[-1] = self.bestEver/self.curEf # relative function eval
    assert np.any(np.isfinite(state) == False) == False, "State not finite {}".format(state)
    return state

  def getReward(self):
    #r = (self.prevEf - self.curEf)/self.initialEf
    #r = (self.prevEf - self.curEf)/self.prevEf
    #r = +np.log(self.prevEf)-np.log(self.curEf)
    #r = -np.log(self.curEf)

    #r = np.log(self.initialEf/self.curBestF)
    r = np.log(self.initialEf/self.curEf)
    assert np.isfinite(r), "Return not finite {}".format(r)

    return r

if __name__ == '__main__':
    objective = ObjectiveFactory(8, noise=0.0, objective="Random")
    objective.reset()
    objective.advance(np.ones(3))
    state = objective.getState()
    reward = objective.getReward()