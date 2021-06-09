import random
import numpy as np
from itertools import product 
import time

from source.fish import *

class swarm:
    def __init__(self, N, alpha=360., speed=1., turningRate=360., seed=42):
        assert alpha > 0., "wrong alpha"
        assert alpha <= 360., "wrong alpha"
        assert turningRate > 0., "wrong turningRate"
        assert turningRate <= 360., "wrong turningRate"
        # number of fish
        self.N = N
        # field of perception
        self.alpha = alpha / 180. * np.pi
        # swimming speed 
        self.speed = speed
        # turning rate
        self.turningRate = turningRate / 180. * np.pi
        # create fish at random locations
        self.fishes = self.randomPlacementNoOverlap( seed )

    """ random placement on a grid """
    def randomPlacementNoOverlap(self, seed):
        # number of gridpoints per dimension for initial placement
        M = int( pow( self.N, 1/3 ) )
        V = M+1
        # grid spacing ~ min distance between fish
        dl = 0.1
        # maximal extent
        L = V*dl
        
        # generate random permutation of [1,..,V]x[1,..,V]x[1,..,V]
        perm = list(product(np.arange(0,V),repeat=3))
        assert self.N < len(perm), "More vertices required to generate random placement"
        random.Random( seed ).shuffle(perm)
    
        # place fish
        fishes = np.empty(shape=(self.N,), dtype=fish)
        for i in range(self.N):
          location = np.array([perm[i][0]*dl, perm[i][1]*dl, perm[i][2]*dl]) - L/2
          fishes[i] = fish(location, self.speed, self.turningRate)
        
        # return array of fish
        return fishes

    """ compute distance and angle matrix """
    def computeStates(self):
        # create containers for distances and angles
        distances = np.full(shape=(self.N,self.N), fill_value=np.inf, dtype=float)
        angles    = np.zeros(shape=(self.N,self.N), dtype=float)
        # iterate over grid and compute angle / distance matrix
        # TODO: use scipy.spatial.distance_matrix
        for i in np.arange(self.N):
          for j in np.arange(self.N):
            if i != j:
              # direction vector and current direction
              u = self.fishes[j].location - self.fishes[i].location
              v = self.fishes[i].curDirection
              # set distance
              distances[i,j] = np.linalg.norm(u)
              # set angle
              cosAngle = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
              angles[i,j] = np.arccos(cosAngle)

        return distances, angles

    ''' according to https://doi.org/10.1006/jtbi.2002.3065 '''
    def move(self):
        anglesMat, distancesMat = self.computeStates()
        for i in np.arange(self.N):
            deviation = anglesMat[i,:]
            distances = distancesMat[i,:]
            visible = abs(deviation) <= ( self.alpha / 2. )

            rRepell  = self.rRepulsion   * ( 1 + fishes[i].epsRepell  )
            rOrient  = self.rOrientation * ( 1 + fishes[i].epsOrient  ) 
            rAttract = self.rAttraction  * ( 1 + fishes[i].epsAttract )

            repellTargets  = self.fishes[(distances < rRepell)]
            orientTargets  = self.fishes[(distances >= rRepell) & (distances < rOrient) & visible]
            attractTargets = self.fishes[(distances >= rOrient) & (distances <= rAttract) & visible]
            self.fishes[i].computeDirection(repellTargets, orientTargets, attractTargets)

        for fish in self.fishes:
            fish.updateDirection()
            fish.updateLocation()

    ''' utility to compute polarisation (~alignement) '''
    def computePolarisation(self):
        polarisationVec = np.zeros(shape=(3,), dtype=float)
        for fish in self.fishes:
            polarisationVec += fish.curDirection
        polarisation = np.linalg.norm(polarisationVec) / self.N
        return polarisation

    ''' utility to compute center of swarm '''
    def computeCenter(self):
        center = np.zeros(shape=(3,), dtype=float)
        for fish in self.fishes:
            center += fish.location
        center /= self.N
        return center

    ''' utility to compute angular momentum (~rotation) '''
    def computeAngularMom(self):
        center = self.computeCenter()
        angularMomentumVec = np.zeros(shape=(3,), dtype=float)
        for fish in self.fishes:
            distance = fish.location-center
            distanceNormal = distance / np.linalg.norm(distance) 
            angularMomentumVecSingle = np.cross(distanceNormal,fish.curDirection)
            angularMomentumVec += angularMomentumVecSingle
        angularMomentum = np.linalg.norm(angularMomentumVec) / self.N
        return angularMomentum
