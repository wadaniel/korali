#!/usr/bin/env python3

##  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

from dataclasses import dataclass
import math
import numpy as np
import scipy.integrate as integrate

def compute_V(bmb: float,
              cmb: float,
              omega: float):
    wc = cmb

    if omega <= wc:
        return bmb / cmb * omega
    else:
        # helper constant:
        a1 = np.sqrt(omega**2 - wc**2)
        a2 = np.arctan(wc / a1)
        period = 2 * np.pi / a1

        def integrand(t):
            theta = 2 * np.arctan(wc / omega - a1 / omega * np.tan(0.5 * a1 * t - a2))
            return np.sin(theta)

        return bmb * integrate.quad(integrand, 0, period)[0] / period


@dataclass
class ABF:
  # parameters of the shape
  bmb: float
  cmb: float

  def velocity(self, omega: float):
    return compute_V(self.bmb, self.cmb, omega)



class Swimmers:
  def __init__(self):
    self.dt = 1
    self.t_max = 100
    self.target_radius = 1
    self.ABFs = [ABF(bmb=1, cmb=1), ABF(bmb=1, cmb=2)]
    self.reset()

  def reset(self):
    self.step = 0
    self.t = 0
    self.positions = np.array([[15., 10.], [-5., 15.]])
    self.prevDistance = self.getSumDistances()

  def isSuccess(self):
    diatances = np.sqrt(np.sum(self.positions**2, axis=1))
    return np.max(diatances) < self.target_radius

  def isOver(self):
    return self.isSuccess() or self.t > self.t_max

  def system(self, t, y, act):
    wx, wy, omega = act
    Vs = [a.velocity(omega) for a in self.ABFs]
    u = np.array([wx, wy]) / np.sqrt(wx**2 + wy**2)
    return np.array([ v * u for v in Vs])


  def getSumDistances(self):
    return np.sum(np.sqrt(np.sum(self.positions**2, axis=1)))

  def advance(self, action):
    self.prevDistance = self.getSumDistances()
    self.action = np.array(action)
    self.positions += self.dt * self.system(self.t, self.positions, action) # Forward Euler is exact here
    self.t += self.dt
    self.step += 1
    return self.isOver()

  def getState(self):
    return self.positions.flatten()

  def getReward(self):
    currDistance = self.getSumDistances()
    return -dt + (self.prevDistance - currDistance)


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  def plot_velocity_curves():
    omega = np.linspace(0, 3, 256)
    ABFs = [ABF(bmb=1, cmb=1), ABF(bmb=1, cmb=2)]
    Vs = np.zeros((len(omega), len(ABFs)))

    for i, a in enumerate(ABFs):
      Vs[:,i] = [a.velocity(w) for w in omega]

    fig, ax = plt.subplots()
    for i in range(len(ABFs)):
      ax.plot(omega, Vs[:,i])
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$V$')
    plt.show()

  def try_simple_action():
    sim = Swimmers()
    action1 = np.array([1, 0, 1])
    action2 = np.array([0, -1, 2])

    over = False

    traj = list()

    while not over:
      sim.advance(action1)
      traj.append(sim.getState())
      over = sim.advance(action2)
      traj.append(sim.getState())

    traj = np.array(traj)
    fig, ax = plt.subplots()
    for i in range(len(sim.ABFs)):
      ax.plot(traj[:,i*2], traj[:,i*2+1])
    target = plt.Circle((0, 0), sim.target_radius, color='r', alpha = 0.3)
    ax.add_artist(target)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.show()

  #plot_velocity_curves()
  try_simple_action()
