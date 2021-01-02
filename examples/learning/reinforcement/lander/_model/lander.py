#!/usr/bin/env python3

##  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

import numpy as np
import math

class Lander:
  def __init__(self):
    self.dt = 0.04
    self.t_max = 8
    self.t = 0.0
    self.g = -0.6
    self.posY = 0.0
    self.posX = 0.0
    self.velY = 0.0
    self.velX = 0.0
    self.step=0

  def reset(self, seed):
    self.posX = np.random.uniform(-0.2, 0.2)
    self.posY = 0.9
    self.velX = np.random.uniform(-0.5, 0.5)
    self.velY = np.random.uniform(-0.2, 0.1)
    self.prevDistance = self.getDistance()
    self.step = 0
    self.t = 0
    
  def isOver(self):
    return self.posY < 0.0

  def getDistance(self):
    return math.sqrt(self.posY*self.posY + self.posX*self.posX)  

  def advance(self, action):
    self.prevDistance = self.getDistance()
  
    self.fX = action[0]
    if (self.fX > 1.0):
      self.fX = 1.0
    elif self.fX < -1.0:
      self.fX = -1.0

    self.fY = action[1]
    if (self.fY > 1.0):
      self.fY = 1.0
    elif self.fY < -1.0:
      self.fY = -1.0

    self.velX = self.velX +  self.fX * self.dt
    self.velY = self.velY +  (self.fY + self.g) * self.dt
    
    self.posX = self.posX +  self.velX * self.dt
    self.posY = self.posY +  self.velY * self.dt
      
    self.t = self.t + self.dt
    self.step = self.step + 1
    
    if self.isOver(): 
      return 1
    else: 
      return 0

  def getState(self):
    state = np.zeros(2)
    state[0] = self.posX # X Position
    state[1] = self.posY # Y Position
    #state[2] = self.velX # X Velocity
    #state[3] = self.velY # Y Velocity
    return state

  def getReward(self):
    currDistance = self.getDistance()
    r = -self.dt / self.t_max
    r += self.prevDistance - currDistance # reward shaping
    if (self.posY <= 0): r = r + 4 - 20*(self.velY*self.velY)
    return r
