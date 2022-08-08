import numpy as np
from numpy.random import normal, uniform
import matplotlib.pyplot as plt

def sample_theta(size_n):
    e = np.sqrt( (1-np.exp(-0.8))/2.88)
    return uniform(0, e, size=(size_n, 1))

def model(theta, d): # G
    res = theta * theta * theta * d * d + theta * np.exp(-np.abs(0.2 - d))
    return res

def data(theta, d, sigma):
    n = theta.shape[0]
    m = model(theta, d)
    epsilon = normal(loc=0, scale=sigma, size=(n, 1))
    return m + epsilon

def likelihood(y, theta, d, sigma): # for the first term
    # for each y compute likelihood with corresponding theta
    mu = model(theta, d)
    res = (1/np.sqrt(2*np.pi*sigma*sigma)) * np.exp(-(y-mu)*(y-mu) / (2*sigma*sigma))
    return res

def likelihood2(y, theta, d, sigma): # for the second term
    # we only have one y and compute the likelihood of that y for all theta
    mu = model(theta, d)
    n = theta.shape[0]
    vec = np.zeros((n, 1))
    for i in range(n):
        res = (1/np.sqrt(2*np.pi*sigma*sigma)) * np.exp(-np.multiply((y-mu[i]), (y-mu[i])) / (2*sigma*sigma))
        vec += res/n
    return vec

def utility(y_vec, theta_vec, d, sigma):
    n = theta_vec.shape[0]
    like = likelihood(y_vec, theta_vec, d, sigma)
    first = np.log(like)

    like2 = likelihood2(y_vec, theta_vec, d, sigma)
    second = np.log(like2)

    total = (1/n) * np.sum(first - second)
    return total



n = 2000
sigma = 1e-2
ds = np.linspace(0, 1, 101)
U = np.zeros(101)



for ind, d in enumerate(ds):
    print(ind)
    thetas = sample_theta(n)
    ys = data(thetas, d, sigma)
    U[ind] = utility(ys, thetas, d, sigma)


plt.figure(1)

plt.plot(ds, U)

plt.savefig('utility.png')


# design space d = [0, 1]

# prior \theta ~ U (0, 1) = p(\theta)

# y(\theta, d) = G(\theta, d) + \epsilon = \theta^3 * d^2 + \theta * exp(-|0.2 - d|) + \epsilon
# => y ~ N (G(\theta, d), sigma^2), where epsilon ~ N (0, sigma^2)


# n_out = number of samples of the data, 
# y^(i) drawn from p(y| \theta = \theta^(i), d) => sample different y for different theta, given d
# the theta are drawn from the prior p(\theta)

# to compute the value of p(y^(i)| \theta^(i), d)


# p(y^(i)|d) is usually not analytic, so we approximate using important sampling
# p(y^(i)|d) = (1/n_in) * \sum_{j=1}^n_in p(y^(i)| \theta^(i, j), d), 
# \theta^(., j) are drawn from the prior p(\theta)


# U = (1/n_out) * \sum_{i=1}^n_out [ \ln(p(y^(i)| \theta = \theta^(i), d)) - \ln((1/n_in) * \sum_{j=1}^n_in p(y^(i)| \theta^(i, j), d))]

# to mitigate cost of nested MC estimator, draw a fresh batch of prior samples {\theta^(k)}_{k=1}^n_out for every d
# use this set for both outer and inner MC sums, i.e. \theta^(i) = \theta^(k) and \theta^(., j) = \theta^(k) 
# n_out = n_in

# we can now compute the utility



# # step 1 : try out the model, it works

# m = model(thetas, 0.5)
# print(thetas.shape)
# print(m.shape)

# plt.figure(1)

# plt.plot(thetas, m, '.') # the points are not sorted, so plotting with lines between data points is wrong
# plt.plot(ds, model(ds, 0.5))

# plt.savefig('model.png')

# # step 2 : generate data with noise, it works

# ys = data(n, thetas, 0.5, sigma)

# plt.figure(2)

# plt.plot(thetas, ys, 'k.')
# plt.plot(ds, model(ds, 0.5))

# plt.savefig('noisy_data.png')

