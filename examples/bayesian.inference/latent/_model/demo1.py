#! /usr/bin/env python3

import numpy as np
from gaussian_mixture import gm

import korali

class generative_model():

    def __init__(self):
        self.mean = np.array([[0,0]])
        sigma2 = 2.
        self.covariance = sigma2 * np.array([np.identity(2)])
        self.weights = np.array( [1] )

        self.rv = gm(self.mean, self.covariance, self.weights)

    def generate_data(self,N):
        return self.rv.rvs(N)[0]


class model():

    def __init__(self,data):
        if data.ndim==1:
            data = np.expand_dims(data, axis=0)

        self.data = data
        self.data_mean = np.mean(data,axis=0)

        self.mu_lower_bound = -20
        self.mu_upper_bound = 20

        self.N = data.shape[0]
        self.dim = data.shape[1]

        self.c0 = self.N * np.log(self.mu_upper_bound-self.mu_lower_bound)
        self.c1 = 0.5 * self.N * self.dim * np.log(2*np.pi)
        self.c2 = 0.5 * self.N * self.dim

        self.latent_means = [1,1]
        self.N_latent = len(self.latent_means)

    def zeta(self,sample):
        sigma2 = sample['Hyperparameters'][0]
        sample['zeta'] = self.c1 + self.c2*np.log(sigma2)

    def S(self,sample):
        mean = np.array( sample['Latent Variables'] )
        sample['S'] = [ - np.sum( np.square( self.data-mean) ) ]

    def phi(self,sample):
        sigma2 = sample['Hyperparameters'][0]
        sample['phi'] = [ 0.5/sigma2 ]

    def A(self,sample):
        sample['A'] = self.c0

    def latent_data_logpdf(self,sample):
        self.A(sample)
        return data_conditional_logpdf(self,sample) + sample['A']

    def data_conditional_logpdf(self,sample):
        self.zeta(sample)
        self.S(sample)
        self.phi(sample)
        return - sample['zeta'] + sample['S'][0]*sample['phi'][0]

    def latent_conditional_logpdf(self, sample, hyperparameters):
        sample['Latent Variables'] = sample['Parameters']
        sample['Hyperparameters'] = hyperparameters
        sample['P(x)'] = self.data_conditional_logpdf(sample)

    def sample_latent(self, sample):
        hyperparameters = sample['Hyperparameters']
        numberSamples = sample['Number Samples']

        mean = self.data_mean
        sigma2 = hyperparameters[0]
        covariance = sigma2*np.identity(self.dim)

        samples = np.random.multivariate_normal(mean, covariance, numberSamples)
        samples = samples / np.sqrt(self.N)
        sample['Samples'] = samples.tolist()

    def sample_latent_mcmc(self, sample, verbosity='Silent', fof=0, cof=0):
        hyperparameters = sample['Hyperparameters']
        numberSamples = sample['Number Samples']

        k = korali.Engine()
        e = korali.Experiment()

        e['Problem']['Type'] = 'Sampling'
        e['Problem']['Probability Function'] = lambda s: self.latent_conditional_logpdf(s,hyperparameters)

        for i in range(self.N_latent):
            e['Variables'][i]['Name'] = 'latent_' + str(i)
            e['Variables'][i]['Initial Mean'] = self.latent_means[i]
            e['Variables'][i]['Initial Standard Deviation'] = 2.0

        e['Solver']['Type'] = 'Sampler/MCMC'
        e['Solver']['Burn In'] = 100
        e['Solver']['Termination Criteria']['Max Samples'] = numberSamples

        e['File Output']['Frequency'] = fof
        e['File Output']['Path'] = '_korali_sampling_results'
        e['Console Output']['Frequency'] = cof
        e['Console Output']['Verbosity'] = verbosity

        k.run(e)

        samples = e['Solver']['Sample Database'][-numberSamples:]
        sample['Samples'] = samples

        self.latent_means = np.sum(np.array(samples),axis=0) / float(numberSamples)

    def check_conditional_pdf(self,model):
        sample = {}
        sample['Hyperparameters'] = [0.5]
        sample['Latent Variables'] = [0.,0.]

        res1 = self.data_conditional_logpdf(sample)
        res2 = np.sum( np.log( model.rv.pdf(self.data) ) )

        assert np.isclose([res1],[res2])
        print('\nExponential Family implementation validated.\n')

    def check_latent_sampling(self):
        sample = {}
        sample['Hyperparameters'] = [1]
        sample['Number Samples'] = 5000
        # self.sample_latent_mcmc(sample,verbosity='Normal',fof=100,cof=100)
        self.sample_latent(sample)


def main():
    gen_model = generative_model()
    data = gen_model.generate_data(10)
    np.savetxt('data1.txt',data)

    m = model(data)
    m.check_conditional_pdf(gen_model)
    m.check_latent_sampling()

if __name__=='__main__':
    main()