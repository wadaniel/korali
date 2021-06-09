import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import vonmises

from source.plotter import *

class fish:
    def __init__(self, location, individualStd=0.1, speed=3, maxAngle=180./180.*np.pi, eqDistance=0.1, potentialStrength=1, potential="Harmonic" ):
        self.location = location
        self.curDirection = self.randUnitDirection()
        self.wishedDirection = self.curDirection
        # individual variation
        self.individualStd = individualStd
        individualNoise = np.zeros(4) #np.random.normal(0.0, self.individualStd, 4)
        # motion parameters
        self.speed       = speed    * ( 1 + individualNoise[0] )
        self.maxAngle    = maxAngle * ( 1 + individualNoise[1] )
        self.dt          = 0.1
        self.sigmaMotion = 0.1
        # potential
        self.potential = potential
        ## parameters for Lennard-Jones potential
        if self.potential == "Lennard-Jones":
            self.epsilon        = potentialStrength * ( 1 + individualNoise[2] ) # max value of reward
            self.sigmaPotential = eqDistance        * ( 1 + individualNoise[3] ) # distance below which reward becomes penality
        ## parameters for harmonic potential
        elif self.potential == "Harmonic":
            self.k = potentialStrength * ( 1 + individualNoise[2] ) # strength of force towards optimal distance
            self.r0 = eqDistance       * ( 1 + individualNoise[3] ) # equilibrium distance
        else: 
            assert 0, print("Please chose a potential that is implemented")

    ''' get uniform random unit vector on sphere '''      
    def randUnitDirection(self):
        vec = np.random.normal(0.,1.,3)
        mag = np.linalg.norm(vec)
        return vec/mag

    ''' according to https://doi.org/10.1006/jtbi.2002.3065 '''
    def computeDirection(self, repellTargets, orientTargets, attractTargets):
        newWishedDirection = np.zeros(3)
        # zone of repulsion - highest priority
        if repellTargets.size > 0:
            for fish in repellTargets:
                diff = fish.location - self.location
                assert np.linalg.norm(diff) > 1e-12, print(diff, "are you satisfying speed*dt<=rRepulsion?")
                assert np.linalg.norm(diff) < 1e12,  print(diff)
                newWishedDirection -= diff/np.linalg.norm(diff)
        else:
            orientDirect = np.zeros(3)
            attractDirect = np.zeros(3)
            # zone of orientation
            if orientTargets.size > 0:
              for fish in orientTargets:
                  orientDirect += fish.curDirection/np.linalg.norm(fish.curDirection)
            # zone of attraction
            if attractTargets.size > 0:
              for fish in attractTargets:
                  diff = fish.location - self.location
                  attractDirect += diff/np.linalg.norm(diff)
            
            newWishedDirection = orientDirect+attractDirect
        
        if np.linalg.norm(newWishedDirection) < 1e-12:
          newWishedDirection = self.curDirection
        
        ## stochastic effect, replicates "spherically wrapped Gaussian distribution"
        # get random unit direction orthogonal to newWishedDirection
        randVector = self.randUnitDirection()
        rotVector = np.cross(newWishedDirection,randVector)
        while np.linalg.norm(rotVector) < 1e-12:
            randVector = self.randUnitDirection()
            rotVector = np.cross(newWishedDirection,randVector)
        rotVector /= np.linalg.norm(rotVector)
        # compute random angle from wrapped Gaussian ~ van Mises distribution
        randAngle = vonmises.rvs(1/self.sigma**2)
        # create rotation
        rotVector *= randAngle
        r = Rotation.from_rotvec(rotVector)
        # apply rotation
        self.wishedDirection = r.apply(newWishedDirection)


    ''' rotate direction of the swimmer ''' 
    def updateDirection(self):
        u = self.curDirection
        v = self.wishedDirection
        assert np.linalg.norm(u) > 1e-12, print(u)
        assert np.linalg.norm(v) > 1e-12, print(v)
        assert np.linalg.norm(u) < 1e12, print(u)
        assert np.linalg.norm(v) < 1e12, print(v)

        cosAngle = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
        angle = np.arccos(cosAngle)
        if angle < self.maxAngle:
            self.curDirection = self.wishedDirection
        else:
            rotVector = np.cross(self.curDirection, self.wishedDirection)
            rotVector /= np.linalg.norm(rotVector)
            rotVector *= maxAngle
            r = Rotation.from_rotvec(rotVector)
            self.curDirection = r.apply(self.curDirection)
        
        # normalize
        self.curDirection /= np.linalg.norm(self.curDirection)

    ''' update the direction according to x += vt ''' 
    def updateLocation(self):
        self.location += self.speed*self.dt*self.curDirection

    ''' reward assumes pair-wise Lennard-Jones potentials ''' 
    def getReward(self, nearestNeighbourDistance ):
        reward = 0.0
        for r in nearestNeighbourDistance:
            # Lennard-Jones potential
            if self.potential == "Lennard-Jones":
                x = self.sigmaPotential / r
                reward -= 4*self.epsilon*( x**12 - x**6 )
            # Harmonic potential
            elif self.potential == "Harmonic":
                reward -= 1/2*self.k*(r-self.r0)**2
        return reward