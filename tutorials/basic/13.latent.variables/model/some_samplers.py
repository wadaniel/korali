
import numpy as np
from scipy.stats import multivariate_normal
import pdb

import korali

class MCMCLatentSampler:
    def __init__(self,   numberLatentVars, numberHyperparams, initialLatentValues,initialHyperparams,
                    zeta_,  S_, phi_, do_sample_discrete_ , min_if_discrete_, max_if_discrete_  ):
        self.zeta_func = zeta_
        self.S_func = S_
        self.phi_func = phi_

        self.sample_discrete = do_sample_discrete_
        self.min_if_discrete = min_if_discrete_
        self.max_if_discrete = max_if_discrete_

        assert (initialLatentValues.size() == numberLatentVars ) ,"number of latent variables should match the dimension of their initial values"
        assert (initialHyperparams.size() == numberHyperparams ) ,"number of hyperparameters should match the dimension of their initial values"

        self.previousSampleMeans = initialLatentValues
        self.currentHyperparameters = initialHyperparams

        self.numberLatent = numberLatentVars
        self.numberHyperparameters = numberHyperparams

    def sampleLatent(self, kSample):
        # /*
        #  * probability to sample from:
        #  * p(d, z | theta) * p(theta) -- that's the (log-)posterior
        #  * - use a "Custom" bayesian problem, with our log-posterior as "Likelihood Model" , but with the current values for theta inserted.
        # */

        hyperparameters = kSample["Hyperparameters"];
        numberSamples = kSample["Number Samples"];
        if (kSample["Number Of Latent Variables"] != self.numberLatent):
            raise ValueError("Implementation error, number of latent variables at initialization does not fit to what was passed as variable")


        # /* Create one sampling experiment to sample all latent variables. After all, the latent vars are correlated /
        #     have a joint distrib.
        #     Todo: does it make sense to re-create these experiments at every E-M step? Otherwise, how
        #         to automatically update the initial mean and the distribution function of the sampling experiments?*/

        k = korali.Engine()
        e = korali.Experiment()

        def probability_function(s):

             if not ( "Parameters" in s):
                raise ValueError("Something is trying to evaluate the likelihood without passing values for the latent variables (= parameters, here) to the sample.\n")

             latent_vars = s["Parameters"] #.get < std::vector < double >> ();
             hparams = hyperparameters

             if (len(latent_vars) != self.numberLatent):
                 raise ValueError("Implementation error, latent variable vector had wrong size")

             if (self.hparams.size() != self.numberHyperparameters):
                 raise ValueError("Implementation error, hyperparameter vector had wrong size");

             s["Latent Variables"] = latent_vars;
             s["Hyperparameters"] = hparams;

             self.S_func(s)
             self.zeta_func(s)
             self.phi_func(s)

            # -> Assume these set: sample["S"], sample["zeta"] and sample["phi"]
             if not ( s.contains("S")):
                 raise KeyError("The specified likelihood model did not assign the value: 'S' to the sample.\n")
             if not ( s.contains("zeta")):
                 raise KeyError("The specified likelihood model did not assign the value: 'zeta' to the sample.\n")
             if not ( s.contains("phi")):
                 raise KeyError("The specified likelihood model did not assign the value: 'phi' to the sample.\n")

             _zetaValue = s["zeta"] #.get < double > ();
             _sValues = s["S"] #.get < std::vector < double >> ();
             _phiValues = s["phi"] #.get < std::vector < double >> ();
             logP_of_x = - _zetaValue + np.inner(_sValues,  _phiValues)
             s["P(x)"] = logP_of_x
        # * * * * *  * * * * *  * * * * * probability_function end

         # // Based on tutorial a2-sampling
        e["Problem"]["Type"] = "Sampling"
        e["Problem"]["Probability Function"] = lambda s : probability_function(s)

        for i in range(self.numberLatent):

            # Defining problem's variables
            varName = "latent_" + str(i)

            e["Variables"][i]["Name"] = varName
            e["Variables"][i]["Initial Mean"] = self.previousSampleMeans[i]

            e["Variables"][i]["Initial Standard Deviation"] = 2.0
            if (self.sample_discrete):
                e["Variables"][i]["Initial Standard Deviation"] = float(self.max_if_discrete - self.min_if_discrete)/10.0
                e["Variables"][i]["Initial Mean"] = self.previousSampleMeans[i]

        # Configuring the MCMC sampler parameters
        e["Solver"]["Type"]  = "MCMC";
        e["Solver"]["Burn In"] = 500;
        e["Solver"]["Termination Criteria"]["Max Samples"] = 5000;

        # Configuring output settings
        e["File Output"]["Frequency"] = 500;
        e["Console Output"]["Frequency"] = 1000;
        e["Console Output"]["Verbosity"] = "Normal";

        # Todo: I don't think a result path is needed (and it'd need a step id in the pathname as well)
        #e["Results"]["Path"] = "setup/results_phase_1/" + "0"*(3 - str(i).length()) +  std:to_string(i);
        k.run(e)

        db = e["Solver"]["Sample Database"] #.get<std::vector<std::vector<double>>>();
        print("Database size: %lu\n", db.size())
        samples = db[-numberSamples : ]
        initial_samples = db[ : numberSamples]


        kSample["Samples"] = samples
        kSample["Initial Samples For Debugging"] = initial_samples

        # set new "previous sample means"
        for i in range(self.numberLatent):
            self.previousSampleMeans[i] = np.sum(samples[:, i]) / float(numberSamples)

class MultimodalGaussianSampler():

    def __init__(self, points_, nDimensions_, nClusters_):
        self.nDimensions = nDimensions_
        self.nClusters = nClusters_
        self.points = np.array(points_)
        self.nPoints = len(points_)

         # uniform_distrib = std::uniform_real_distribution<>(0.0, 1.0);

    def sampleLatent(self, k):
        # /* Sample assignments for each point separately, proportional to the resulting data likelihood, p(point, assignment | hyperparams) */

        hyperparameters = k["Hyperparameters"]
        nSamples = k["Number Samples"]
        if (k["Number Of Latent Variables"] != self.nPoints):
           raise BaseException("Implementation error, number of latent variables at initialization does not fit to what was passed as variable")

        # get sigma
        sigma = hyperparameters[self.nClusters * self.nDimensions]

        # get vector of means
        mus = np.zeros((self.nClusters, self.nDimensions))
        for i in range(self.nClusters):
            mu = hyperparameters[i * self.nDimensions : (i + 1) * self.nDimensions]
            if(len(mu) != self.nDimensions):
                raise BaseException("Implementation error, dimensions did not match")
            mus[i] = mu


        # we sample in another order, will later transpose
        samples_transpose = np.zeros((self.nPoints, nSamples))
        for i in range(self.nPoints):

            # First, get probabilities for each cluster
            probabilities = np.zeros((self.nClusters,))
            for j in range(self.nClusters):
                mu = mus[j]
                cov = sigma ** 2 * np.eye(len(mu))
                mv = multivariate_normal(mean=mu, cov=cov)
                p = mv.pdf(self.points[i])
                probabilities[j] = p

            # normalize probabilities to one
            probabilities = probabilities / np.linalg.norm(probabilities)

            pointwise_samples = np.zeros((nSamples,), int)
            for j in range(nSamples):
                # sample
                u = np.random.uniform(0,1,1)
                sum = 0
                for c_idx in range(self.nClusters):
                    sum = sum + probabilities[c_idx]
                    if (u < sum):
                        # cluster c_idx has been chosen!
                        pointwise_samples[j] = c_idx
                        break
            samples_transpose[i] = pointwise_samples

        samples = np.transpose(samples_transpose)
        k["Samples"] = samples.tolist()


