#! /usr/bin/env python3

import numpy as np
import numpy.matlib
from scipy import linalg
from gaussian_mixture import gm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



m = np.array([1,2])
s = np.array([[0.5,0.],[0.,0.5]])
y = np.array([0.1,0.2])
c = linalg.inv(s)

res = -np.log(2*np.pi)
res += -0.5 * np.linalg.slogdet(s)[1]
res += -0.5*(y.T-m.T).dot(c.dot(y-m))
print(res)

rv = multivariate_normal(m,s)
res = rv.logpdf(y)
print(res)
