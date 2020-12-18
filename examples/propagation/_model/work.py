#!/usr/bin/env python3
import os
import sys
import numpy as np


def prepareOutputDir():
    if os.path.isdir('_executor_output') == False:
        os.mkdir('_executor_output')


def put_normal_rnds(theta, Ns, fileName):
    mu = theta['Parameters'][0]
    var = theta['Parameters'][1]

    y = np.random.normal(mu, var, Ns)

    if os.path.isdir('_executor_output') == True:
        f = open('_executor_output/{0}'.format(fileName), 'a+')
        np.savetxt(f, np.transpose(y))
        f.close()

    else:
        sys.exit(
            'put_normal_rnds: dir \'_executor_output\' does not exist! exit..')


# This is a linear regression model with two params (slope and intercept)
def model(s, X):
    a = s['Parameters'][0]
    b = s['Parameters'][1]
    sig = s['Parameters'][2]

    s['Reference Evaluations'] = []
    s['Standard Deviation'] = []
    for x in X:
        s['Reference Evaluations'] += [a * x + b]
        s['Standard Deviation'] += [sig]


def model_propagation(s, X):
    a = s['Parameters'][0]
    b = s['Parameters'][1]

    s['sigma'] = s['Parameters'][2]
    s['X'] = X.tolist()
    s['Evaluations'] = []
    for x in X:
        s['Evaluations'] += [a * x + b]


def getReferenceData():
    y = []
    y.append(3.21)
    y.append(4.14)
    y.append(4.94)
    y.append(6.06)
    y.append(6.84)
    return y


def getReferencePoints():
    x = []
    x.append(1.0)
    x.append(2.0)
    x.append(3.0)
    x.append(4.0)
    x.append(5.0)
    return x
