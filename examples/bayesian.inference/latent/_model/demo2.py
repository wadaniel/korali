#! /usr/bin/env python3

import numpy as np
import numpy.matlib
from gaussian_mixture import gm
import matplotlib.pyplot as plt

import korali

class generative_model():

    def __init__(self):
        self.mean = np.array( [ [0.,0.],
                                [2.,3.] ] )
        cov = [[0.5,0.],[0.,0.5]]
        self.covariance = np.array( [ cov, cov ] )
        self.weights = np.array( [1,1] )

        self.rv = gm(self.mean, self.covariance, self.weights)

    def generate_data(self,N):
        return self.rv.rvs(N)

    def plot_data(self, data, cluster):

        x1 = np.amin( data[:,0] )
        x2 = np.amax( data[:,0] )
        y1 = np.amin( data[:,1] )
        y2 = np.amax( data[:,1] )
        dx = x2-x1
        dy = y2-y1

        x1 = x1 - 0.1*dx
        x2 = x2 + 0.1*dx
        y1 = y1 - 0.1*dy
        y2 = y2 + 0.1*dy

        x = np.linspace(x1,x2,100)
        y = np.linspace(y1,y2,100)
        x,y=np.meshgrid(x,y)
        p = np.dstack((x,y))
        z = self.rv.pdf(p)

        fig, ax = plt.subplots(1, 1)
        ax.contour(x, y, z, alpha=0.3)

        for k in range(2):
            p = data[np.where(cluster==k)]
            ax.scatter( p[:,0], p[:,1], s=40 )

        plt.show()


class model():

    def __init__(self, data, nClusters=3):
        if data.ndim==1:
            data = np.expand_dims(data, axis=0)

        self.data = data

        self.N = data.shape[0]
        self.dim = data.shape[1]
        self.nClusters = nClusters

        self.c0 = 0.5 * self.N * self.dim * np.log(2*np.pi)
        self.c1 = 0.5 * self.N * self.dim


        self.N_latent = self.N

        self.w_pos = np.arange(0,self.nClusters)
        x = self.nClusters + np.arange( 0, self.nClusters*self.dim)
        self.m_pos = np.reshape(x,(self.nClusters,self.dim))
        y= self.nClusters*(self.dim+1) + np.arange( 0, self.nClusters*self.dim*self.dim)
        self.s_pos = np.reshape(y,(self.nClusters,self.dim,self.dim))

        self.YY = np.zeros((self.N,self.dim**2))
        for k in range(self.N):
            y = np.array( [ data[k] ] )
            z = y * y.T
            self.YY[k] = z.flatten()

    def zeta(self,sample):
        sample['zeta'] = 0

    def S(self,sample):
        z = np.array( sample['Latent Variables'] )

        M = self.nClusters*( 3 + self.dim + self.dim**2 )
        S = np.zeros((self.N_latent,M))

        m0 = np.zeros((self.nClusters))
        m1 = np.zeros((self.nClusters,self.dim))
        m2 = np.zeros((self.nClusters,self.dim**2))

        for i in range(self.N):
            k = z[i]

            m0[k] = 1.
            e = m0
            m0[k] = 0.

            m1[k] = self.data[k]
            y1 = m1.flatten()
            m1[k] = 0

            m2[k] = self.YY[k]
            y2 = m2.flatten()
            m2[k] = 0

            S[i] = np.concatenate((e,y1,y2,e,e))

        sample['S'] = S.flatten().tolist()

    def phi(self,sample):
        theta = np.array( sample['Hyperparameters'] )
        w = theta[self.w_pos]
        m = theta[self.m_pos]
        s = theta[self.s_pos]

        C = np.zeros(s.shape)
        logdet = np.zeros((self.nClusters,))
        Cm = np.zeros((self.nClusters,self.dim))
        mCm = np.zeros((self.nClusters,))
        for k in range(self.nClusters):
            C[k] = np.linalg.cholesky(s[k])
            sign, logdet[k] = np.linalg.slogdet(s[k])
            Cm[k] = C[k].dot(m[k])
            mCm[k] = Cm[k].dot(m[k])
        logw = np.log(w)

        phi = np.concatenate(( logdet.flatten(),
                                -0.5*C.flatten(),
                                Cm.flatten(),
                                mCm.flatten(),
                                -logw.flatten() ))

        phi = np.matlib.repmat(phi,self.N,1)

        sample['phi'] = phi.flatten().tolist()

    def A(self,sample):
        sample['A'] = self.c0

    def check(self,cluster):
        sample={}
        sample['Latent Variables'] = cluster.tolist()

        w = np.full((self.nClusters),1./self.nClusters)

        m = np.zeros((self.nClusters,self.dim))
        for k in range(self.dim):
            min = np.amin(self.data[:,k])
            max = np.amax(self.data[:,k])
            m[:,k] = np.random.uniform(min,max,self.nClusters)

        s = np.zeros((self.nClusters,self.dim**2))
        for k in range(self.nClusters):
            s[k] = np.identity(self.dim).flatten()

        sample['Hyperparameters'] = np.concatenate( (w,m.flatten(),s.flatten()) ).tolist()

        self.S(sample)
        self.phi(sample)
        print( self.data_conditional_logpdf(sample) )



#     def latent_data_logpdf(self,sample):
#         self.A(sample)
#         return data_conditional_logpdf(self,sample) + sample['A']

    def data_conditional_logpdf(self,sample):
        self.zeta(sample)
        self.S(sample)
        self.phi(sample)
        
        return - sample['zeta'] + numpy.dot( sample['S'], sample['phi'] )
#
#     def latent_conditional_logpdf(self, sample, hyperparameters):
#         sample['Latent Variables'] = sample['Parameters']
#         sample['Hyperparameters'] = hyperparameters
#         sample['P(x)'] = self.data_conditional_logpdf(sample)
#
#     def sampleLatent(self, sample, verbosity='Silent', fof=0, cof=0):
#         hyperparameters = sample['Hyperparameters']
#         numberSamples = sample['Number Samples']
#
#         k = korali.Engine()
#         e = korali.Experiment()
#
#         e['Problem']['Type'] = 'Sampling'
#         e['Problem']['Probability Function'] = lambda s: self.latent_conditional_logpdf(s,hyperparameters)
#
#         for i in range(self.N_latent):
#             e['Variables'][i]['Name'] = 'latent_' + str(i)
#             e['Variables'][i]['Initial Mean'] = self.latent_means[i]
#             e['Variables'][i]['Initial Standard Deviation'] = 2.0
#
#         e['Solver']['Type'] = 'Sampler/MCMC'
#         e['Solver']['Burn In'] = 100
#         e['Solver']['Termination Criteria']['Max Samples'] = numberSamples
#
#         e['File Output']['Frequency'] = fof
#         e['File Output']['Path'] = '_korali_sampling_results'
#         e['Console Output']['Frequency'] = cof
#         e['Console Output']['Verbosity'] = verbosity
#
#         k.run(e)
#
#         samples = e['Solver']['Sample Database'][-numberSamples:]
#
#         sample['Samples'] = samples
#
#         self.latent_means = np.sum(np.array(samples),axis=0) / float(numberSamples)
#
#     def check_conditional_pdf(self,model):
#         sample = {}
#         sample['Hyperparameters'] = [0.5]
#         sample['Latent Variables'] = [0.,0.]
#
#         res1 = self.data_conditional_logpdf(sample)
#         res2 = np.sum( np.log( model.rv.pdf(self.data) ) )
#
#         assert np.isclose([res1],[res2])
#         print('\nExponential Family implementation validated.\n')
#
#     def check_latent_sampling(self):
#         sample = {}
#         sample['Hyperparameters'] = [1]
#         sample['Number Samples'] = 5000
#         self.sampleLatent(sample,verbosity='Normal',fof=100,cof=100)


def main():
    gen_model = generative_model()
    data, clusters = gen_model.generate_data(2)
    np.savetxt('data2.txt',data)

    # gen_model.plot_data(data,cluster)

    m = model(data)
    m.check(clusters)
    # m.check_conditional_pdf(gen_model)
    # m.check_latent_sampling()

if __name__=='__main__':
    main()