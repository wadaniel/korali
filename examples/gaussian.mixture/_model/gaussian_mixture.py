#! /usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class gm():

    def __init__(self, mean, covariance, weights):

        n = []
        n.append(mean.shape[1])
        n.append(covariance.shape[1])
        n.append(covariance.shape[2])

        assert( all( x==n[0] for x in n) )

        n = []
        n.append(mean.shape[0])
        n.append(covariance.shape[0])
        n.append(weights.shape[0])

        assert( all( x==n[0] for x in n) )

        self.mean = mean
        self.covariance = covariance
        self.weights = weights / np.sum(weights)
        self.N = mean.shape[0]
        self.Nd = mean.shape[1]

        self.rv = [ multivariate_normal(mean=self.mean[k],cov=self.covariance[k])
                        for k in range(self.N) ]

    def pdf(self, x):
        s = 0
        for k in range(self.N):
            s += self.weights[k]*self.rv[k].pdf(x)
        return s

    def rvs(self, size):
        choice = np.random.choice(self.N, size, p=self.weights)
        samples = np.zeros((size,self.Nd))
        for k in range(size):
            samples[k] = self.rv[choice[k]].rvs()
        return samples, choice


def example_1d( plot=False ):
    mean = np.array( [  [-4 ],
                        [ 0 ],
                        [ 4 ] ] )
    s = np.array( [ [0.1] ])

    N = mean.shape[0]
    Nd = mean.shape[1]

    covariance = np.zeros((N,Nd,Nd))
    for k in range(N):
        covariance[k] = s

    weights = np.array( [1,2,1] )

    if plot==True:
        g = gm(mean,covariance,weights)
        r,_ = g.rvs(100)
        x = np.linspace(-6,6,1000)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x,g.pdf(x))
        ax.hist(r, bins=40, density=True, color='red', alpha=0.2)
        plt.show()

    return mean, covariance, weights


def example_2d( plot=False ):
    mean = np.array( [  [-2,-2],
                        [ 0, 0],
                        [ 2, 2] ] )

    N = mean.shape[0]
    Nd = mean.shape[1]

    covariance = np.zeros((N,Nd,Nd))
    covariance[0] = np.array( [ [.2,0], [0,.2] ])
    covariance[1] = np.array( [ [.5,0], [0,.5] ])
    covariance[2] = np.array( [ [.3,0.1], [0.1,.2] ])

    weights = np.array( [1,2,1] )

    if plot==True:
        g = gm(mean,covariance,weights)
        r, cluster = g.rvs(100)

        x1 = np.amin( r[:,0] )
        x2 = np.amax( r[:,0] )
        y1 = np.amin( r[:,1] )
        y2 = np.amax( r[:,1] )
        dx = x2-x1
        dy = y2-y1

        x1 = x1 - 0.1*dx
        x2 = x2 + 0.1*dx
        y1 = y1 - 0.1*dy
        y2 = y2 + 0.1*dy

        x = np.linspace(x1,x2,100)
        y = np.linspace(y1,y2,100)
        x, y = np.meshgrid(x,y)
        p = np.dstack( (x,y) )
        z = g.pdf(p)

        fig, ax = plt.subplots(1, 1)
        ax.contour(x, y, z, alpha=0.3)

        for k in range(mean.shape[0]):
            p = r[np.where(cluster==k)]
            ax.scatter( p[:,0], p[:,1], s=40 )

        plt.show()

    return mean, covariance, weights


def main():
    example_1d(plot=True)


if __name__ == '__main__':
    main()
