#! /usr/bin/env python3

import numpy as np
import numpy.matlib
from scipy import linalg
from gaussian_mixture import gm
import matplotlib.pyplot as plt

import korali

from scipy.stats import multivariate_normal

class generative_model():

    def __init__(self):
        self.mean = np.array( [ [1.,2.],
                                [3.,4.] ] )
        cov = [[0.5,0.],[0.,0.5]]
        self.covariance = np.array( [ cov, cov ] )
        self.weights = np.array( [1,1] )

        self.gm = gm(self.mean, self.covariance, self.weights)

    def generate_data(self,N):
        return self.gm.rvs(N)

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
        z = self.gm.pdf(p)

        fig, ax = plt.subplots(1, 1)
        ax.contour(x, y, z, alpha=0.3)

        for k in range(2):
            p = data[np.where(cluster==k)]
            ax.scatter( p[:,0], p[:,1], s=40 )

        plt.show()


class model():

    def __init__(self, data, nClusters=2):

        if data.ndim==1:
            data = np.expand_dims(data, axis=0)

        self.data = data

        self.N = data.shape[0]
        self.dim = data.shape[1]
        self.nClusters = nClusters

        self.c0 = 0.5 * self.dim * np.log(2*np.pi)
        self.c1 = self.N * self.c0

        self.nLatent = self.N

        self.w_pos = np.arange(0,self.nClusters)
        x = self.nClusters + np.arange( 0, self.nClusters*self.dim)

        self.m_pos = np.reshape(x, (self.nClusters,self.dim) )
        self.nHalf = int( self.dim*(self.dim+1)/2 )

        index = np.zeros((self.dim,self.dim))
        i,j = np.tril_indices(self.dim)
        index[i,j] = range(self.nHalf)
        index[j,i] = index[i,j]
        self.s_pos = np.zeros((self.nClusters,self.dim,self.dim))
        for k in range(self.nClusters):
            self.s_pos[k] = self.nClusters*(self.dim+1) + k*self.nHalf + index
        self.s_pos = self.s_pos.astype(int)

        self.sizeS_per_variable = self.nClusters*( 3 + self.dim + self.dim**2 )
        self.sizeS = self.nLatent * self.sizeS_per_variable

        self.YY = np.zeros((self.N,self.dim**2))
        for k in range(self.N):
            y = np.array( [ data[k] ] )
            z = y * y.T
            self.YY[k] = z.flatten()

    def hyperparameters_to_matrix(self,theta):
        w = theta[self.w_pos]
        m = theta[self.m_pos]
        s = theta[self.s_pos]
        return w, m, s

    def zeta(self,sample,flatten=True):
        sample['zeta'] = 0

    def S(self, sample, flatten=True):
        z = np.array( sample['Latent Variables'] )
        S = np.zeros((self.nLatent,self.sizeS_per_variable))

        m0 = np.zeros((self.nClusters))
        m1 = np.zeros((self.nClusters,self.dim**2))
        m2 = np.zeros((self.nClusters,self.dim))

        for i in range(self.N):
            k = z[i].astype(int)

            m0[:] = 0.
            m0[k] = 1.
            e = m0

            m1[:] = 0.
            m1[k] = self.YY[i]
            y1 = m1.flatten()

            m2[:] = 0.
            m2[k] = self.data[i]
            y2 = m2.flatten()

            S[i] = np.concatenate(( e, y1, y2, e, e ))

        if flatten==True:
            sample['S'] = S.flatten().tolist()
        else:
            sample['S'] = S.tolist()

    def phi(self, sample, flatten=True):
        theta = np.array( sample['Hyperparameters'] )

        w, m, s = self.hyperparameters_to_matrix(theta)

        C = np.zeros(s.shape)
        logdet = np.zeros((self.nClusters,))
        Cm = np.zeros((self.nClusters,self.dim))
        mCm = np.zeros((self.nClusters,))
        for k in range(self.nClusters):
            C[k] = linalg.inv(s[k])
            logdet[k] = np.linalg.slogdet(s[k])[1]
            Cm[k] = C[k].dot(m[k])
            mCm[k] = Cm[k].dot(m[k])
        logw = np.log(w)

        phi = np.concatenate(( -0.5*logdet.flatten(),
                                -0.5*C.flatten(),
                                Cm.flatten(),
                                -0.5*mCm.flatten(),
                                -logw.flatten(), ))

        phi = np.matlib.repmat(phi,self.N,1)

        if flatten==True:
            sample['phi'] = phi.flatten().tolist()
        else:
            sample['phi'] = phi.tolist()

    def A(self, sample, flatten=True):
        if flatten==True:
            sample['A'] = -self.c1
        else:
            sample['A'] = np.full((self.N,),-self.c0).tolist()

    def latent_data_logpdf(self, sample, flatten=True):
        self.zeta(sample,flatten)
        self.S(sample,flatten)
        self.phi(sample,flatten)
        self.A(sample,flatten)

        if flatten==True:
            return - sample['zeta'] + np.dot( sample['S'], sample['phi'] ) + sample['A']
        else:
            res = - sample['zeta'] +  np.einsum('ij,ij->i', sample['S'], sample['phi'] ) + sample['A']
            return res

    def sample_latent(self, sample):
        Ns = sample['Number Samples']

        prob = np.zeros((self.N,self.nClusters))
        for i in range(self.nClusters):
            sample['Latent Variables'] = np.full((self.N,),i).tolist()
            prob[:,i] = np.exp( self.latent_data_logpdf(sample,flatten=False) )
        prob /= prob.sum(axis=1, keepdims=True)

        samples = np.zeros((Ns,self.nLatent))
        for i in range(self.N):
            samples[:,i] = np.random.choice(self.nClusters, Ns, p=prob[i,:]).astype(float)

        sample['Samples'] = samples.tolist()

    def check(self, gen_model, cluster):
        sample={}
        sample['Latent Variables'] = cluster.tolist()
        i,j = np.tril_indices(self.dim)
        sample['Hyperparameters'] = np.concatenate( ( gen_model.weights,
                                                      gen_model.mean.flatten(),
                                                      gen_model.covariance[:,i,j].flatten()) ).tolist()
        res1 = self.latent_data_logpdf(sample)
        print('Inner product logp: ',res1)
        res2 = 0
        for k in range(self.N):
            res2 += np.log( gen_model.gm.rv[cluster[k]].pdf(self.data[k]) )
            res2 += np.log( gen_model.weights[cluster[k]] )

        print('Numpy log pdf:      ',res2)

        assert np.isclose([res1],[res2]), 'Validation of latent_data_logpdf failled.'
        print('\nLog pdf validated\n')

        sample['Number Samples'] = 100
        self.sample_latent(sample)
        # print(np.mean(sample['Samples'],axis=0))
        # print(cluster)

def main():
    gen_model = generative_model()
    data, cluster = gen_model.generate_data(10)
    np.savetxt('data2.txt',data)

    # gen_model.plot_data(data,cluster)

    m = model(data,nClusters=2)
    m.check(gen_model,cluster)

if __name__=='__main__':
    main()