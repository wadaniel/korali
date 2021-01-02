#!/usr/bin/env python3

##  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

import numpy as np

class Lander:
  def __init__(self):
    self.dt = 0.04
    self.t = 0.0
    self.g = -0.3
    self.posY = 0.0
    self.posX = 0.0
    self.velY = 0.0
    self.velX = 0.0
    self.step=0

  def reset(self):
    self.posX = np.random.uniform(-0.2, 0.2)
    self.posY = np.random.uniform(0.8, 0.9)
    self.velX = np.random.uniform(-0.1, 0.1)
    self.velY = np.random.uniform(-0.1, 0.1)
    self.step = 0
    self.t = 0

  def isOver(self):
    return self.posY < 0.0

  def advance(self, action):
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
    state = np.zeros(4)
    state[0] = self.posX # X Position
    state[1] = self.posY # Y Position
    state[2] = self.velX # X Velocity
    state[3] = self.velY # Y Velocity
    return state

  def getReward(self):
    reward = -abs(self.posX) -self.posY # We want the lander to be centered on X as much as possible
    if (self.posY <= 0): reward = reward + 100 - 1000*(self.velY*self.velY)
    return reward
