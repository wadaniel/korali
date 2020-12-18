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


def example_1d():
    mean = np.array( [  [-4 ],
                        [ 0 ],
                        [ 4 ] ] )
    s = np.array( [ [1] ])

    N = mean.shape[0]
    Nd = mean.shape[1]

    covariance = np.zeros((N,Nd,Nd))
    for k in range(N):
        covariance[k] = s

    weights = np.array( [1,2,1] )

    return mean, covariance, weights


def example_2d():
    mean = np.array( [  [-2,-2],
                        [ 0, 0],
                        [ 2, 2] ] )
    s = np.array( [ [1,0],
                    [0,1] ])

    N = mean.shape[0]
    Nd = mean.shape[1]

    covariance = np.zeros((N,Nd,Nd))
    for k in range(N):
        covariance[k] = s

    weights = np.array( [1,1,1] )

    return mean, covariance, weights


def main():
    mean, covariance, weights = example_1d();
    g = gm(mean,covariance,weights)
    r = g.rvs(1000)

    x = np.linspace(-6,6,1000)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x,g.pdf(x))
    ax.hist(r, bins=40, density=True, color='red', alpha=0.2)
    plt.show()


if __name__ == '__main__':
    main()
