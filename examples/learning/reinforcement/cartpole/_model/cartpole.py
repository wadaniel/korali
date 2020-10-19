#!/usr/bin/env python3

##  This code was taken and adapted from:
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##  https://github.com/cselab/smarties/blob/master/apps/cart_pole_py/exec.py

import math
import numpy as np, sys
from scipy.integrate import ode

class CartPole:
  def __init__(self):
    self.dt = 0.02
    self.step=0
    self.u = np.asarray([0, 0, 0, 0])
    self.F=0
    self.t=0
    self.ODE = ode(self.system).set_integrator('dopri5')

  def reset(self):
    self.u = np.random.uniform(-0.01, 0.01, 4)
    self.step = 0
    self.F = 0
    self.t = 0

  def isFailed(self):
    return abs(self.u[0])>2.4 or  abs(self.u[2])>np.pi/15

  def isOver(self): # is episode over
    return self.isFailed()

  def isTruncated(self):  # check that cause for termination is time limits
    return (abs(self.u[0])<2.4 and abs(self.u[2])<np.pi/15)

  @staticmethod
  def system(t, y, act): #dynamics function
    #print(t,y,act) sys.stdout.flush()
    mp, mc, l, g = 0.1, 1, 0.5, 9.81
    x, v, a, w = y[0], y[1], y[2], y[3]
    cosy, siny = np.cos(a), np.sin(a)
    #const double fac1 = 1./(mc + mp * siny*siny);  #swingup
    #const double fac2 = fac1/l;
    #res.y2 = fac1*(F + mp*siny*(l*w*w + g*cosy));
    #res.y4 = fac2*(-F*cosy -mp*l*w*w*cosy*siny -(mc+mp)*g*siny);
    totMass = mp + mc
    fac2 = l*(4./3. - mp*cosy*cosy/totMass)
    F1 = act + mp*l*w*w*siny
    wdot = (g*siny - F1*cosy/totMass)/fac2
    vdot = (F1 - mp*l*wdot*cosy)/totMass
    return [v, vdot, w, wdot]

  def advance(self, action):
    self.F = action[0]
    self.ODE.set_initial_value(self.u, self.t).set_f_params(self.F)
    self.u = self.ODE.integrate(self.t + self.dt)
    self.t = self.t + self.dt
    self.step = self.step + 1
    if self.isOver(): return 1
    else: return 0

  def getState(self):
    state = np.zeros(5)
    state[0] = np.copy(self.u[0]) # Position
    state[1] = np.copy(self.u[1]) # Velocity
    state[2] = np.copy(self.u[3]) # Omega
    state[3] = np.cos(self.u[2]) # Cos(Angle) 
    state[4] = np.sin(self.u[2]) # Sin(Angle)
    return state

  def getReward(self):
    #double angle = std::fmod(u.y3, 2*M_PI); #swingup
    #angle = angle<0 ? angle+2*M_PI : angle;
    #return fabs(angle-M_PI)<M_PI/6 ? 1 : 0;
    return 1.0 - 1.0 * self.isFailed();

