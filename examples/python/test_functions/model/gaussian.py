#!/usr/bin/env python
import math
import numpy as np
from scipy.stats import multivariate_normal

def gaussian(x):

    dim   = x.getParameterCount()
    mu    = [-3.0]*dim
    Sigma = np.diag([5.0]*dim)
    
    var = multivariate_normal(mean=mu, cov=Sigma)
    result = var.pdf(x.getParameters())
    
    x.addResult(result)
