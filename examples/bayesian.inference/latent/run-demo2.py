#!/usr/bin/env python3

import sys
sys.path.append('./_model')
import numpy as np

from demo2 import *
import korali

def main():

    # the generative model is a Gaussian mixture
    gen_model = generative_model()
    data, cluster = gen_model.generate_data(10)
    m = model(data,nClusters=2)

    e = korali.Experiment()

    e['Problem']['Type'] = 'Bayesian/Latent/Exponential'
    e['Problem']['S Of Likelihood Model'] = m.S
    e['Problem']['Zeta Of Likelihood Model'] = m.zeta
    e['Problem']['Phi Of Likelihood Model'] = m.phi
    e['Problem']['S Dimension'] = 17 #m.sizeS
    e['Problem']['Latent Variable Sampler'] = lambda sample: m.sample_latent(sample)

    e['Solver']['Type'] = 'SAEM'
    e['Solver']['Number Samples Per Step'] = 10
    e['Solver']['Termination Criteria']['Max Generations'] = 2000

    cnt = 0
    for k in range(m.nClusters):
        e['Variables'][cnt]['Bayesian Type'] = 'Hyperparameter'
        e['Variables'][cnt]['Name'] = 'w_' + str(k)
        e['Variables'][cnt]['Prior Distribution'] = 'Uniform Weights'
        e['Variables'][cnt]['Initial Value'] = 1/m.nClusters
        e['Variables'][cnt]['Lower Bound'] = 0.
        e['Variables'][cnt]['Upper Bound'] = 1.
        cnt += 1
    for k in range(m.nClusters):
        for i in range(m.dim):
            e['Variables'][cnt]['Bayesian Type'] = 'Hyperparameter'
            e['Variables'][cnt]['Name'] = 'm_' + str(k) + ',' + str(i)
            e['Variables'][cnt]['Prior Distribution'] = 'Uniform Mean'
            e['Variables'][cnt]['Initial Value'] = k
            e['Variables'][cnt]['Lower Bound'] = -10.
            e['Variables'][cnt]['Upper Bound'] =  10.
            cnt += 1
    for k in range(m.nClusters):
        for i in range(m.dim):
            for j in range(i+1):
                e['Variables'][cnt]['Bayesian Type'] = 'Hyperparameter'
                e['Variables'][cnt]['Name'] = 's_' + str(k) + ',' + str(i) + ',' + str(j)
                e['Variables'][cnt]['Prior Distribution'] = 'Uniform Cov'
                if(i==j):
                    e['Variables'][cnt]['Initial Value'] = 2.
                else:
                    e['Variables'][cnt]['Initial Value'] = 0.
                e['Variables'][cnt]['Lower Bound'] = 0.
                e['Variables'][cnt]['Upper Bound'] =  10.
                cnt += 1

    for k in range(data.shape[0]):
        e['Variables'][cnt]['Name'] = 'z_' + str(k)
        e['Variables'][cnt]['Bayesian Type'] = 'Latent'
        e['Variables'][cnt]['Prior Distribution'] = 'Uniform Zeta'
        e['Variables'][cnt]['Initial Value'] = np.random.choice(m.nClusters)
        cnt += 1

    e['Distributions'][0]['Name'] = 'Uniform Weights'
    e['Distributions'][0]['Type'] = 'Univariate/Uniform'
    e['Distributions'][0]['Minimum'] = 0
    e['Distributions'][0]['Maximum'] = 5

    e['Distributions'][1]['Name'] = 'Uniform Mean'
    e['Distributions'][1]['Type'] = 'Univariate/Uniform'
    e['Distributions'][1]['Minimum'] = -5
    e['Distributions'][1]['Maximum'] = 5

    e['Distributions'][2]['Name'] = 'Uniform Cov'
    e['Distributions'][2]['Type'] = 'Univariate/Uniform'
    e['Distributions'][2]['Minimum'] = -5
    e['Distributions'][2]['Maximum'] = 5

    e['Distributions'][3]['Name'] = 'Uniform Zeta'
    e['Distributions'][3]['Type'] = 'Univariate/Uniform'
    e['Distributions'][3]['Minimum'] = 0
    e['Distributions'][3]['Maximum'] = 1

    e['File Output']['Path'] = '_korali_result_saem'
    e['Random Seed'] = 0xC0FFEE

    engine = korali.Engine()
    engine.run(e)


if __name__ == '__main__':
    # import sys, trace
    # sys.stdout = sys.stderr
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.runfunc(main)
    main()
