#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def plotVariables(genList):
  numgens = len(genList)

  lastGen = 0
  generations = []
  for i in genList:
    generations.append(genList[i]["Current Generation"])
    if genList[i]['Current Generation'] > lastGen:
      lastGen = genList[i]['Current Generation']

  some_gen = list(genList.keys())[0]
  numIndividuals = genList[some_gen]["Solver"]["Number Individuals"]
  numLatentSpaceDims = genList[some_gen]["Solver"]["Latent Space Dimensions"]
  # numLatent = numIndividuals * numLatentSpaceDims
  # numHyper =  numLatentSpaceDims * (1 + numLatentSpaceDims)
  numChains = genList[some_gen]["Solver"]["Number Samples Per Step"]
  N1 = genList[some_gen]["Solver"]["N1"]
  N2 = genList[some_gen]["Solver"]["N2"]
  N3 = genList[some_gen]["Solver"]["N3"]
  nOuter = genList[some_gen]["Solver"]["MCMC Outer Steps"]
  nInner = N1 + N2 + N3
  sample_means = np.zeros((numgens, numIndividuals, numLatentSpaceDims))
  sample_sdevs = np.zeros((numgens, numIndividuals, numLatentSpaceDims))
  hyperparams_mean = np.zeros((numgens, numLatentSpaceDims))
  hyperparams_cov = np.zeros((numgens, numLatentSpaceDims, numLatentSpaceDims))
  gammas = np.zeros((numgens))
  total_logprobab = np.zeros((numgens))
  total_logllh = np.zeros((numgens))
  total_logprior = np.zeros((numgens))
  total_logprior_check = np.zeros((numgens))

  colors = get_cmap(numIndividuals)
  ncols = 3
  nrows = max(numLatentSpaceDims, 3)

  # determine which are latent, vs. hyperparameters
  latentVars = []
  hyperVars = []
  for var_dict in genList[some_gen]["Variables"]:
    if var_dict["Bayesian Type"] == "Hyperparameter":
      hyperVars.append(var_dict)
    else:
      latentVars.append(var_dict)
  del hyperVars  # To prevent accidentially using this; can be empty. latentVars are also not all latent variables.

  log_all_samples = False
  for idx, i in enumerate(genList):
    sample_means[idx, ...] = genList[i]["Solver"]["Current Sample Means"]
    sample_sdevs[
        idx, ...] = genList[i]["Solver"]["Current Sample Standard Deviations"]
    hyperparams_mean[idx,
                     ...] = genList[i]["Solver"]["Current Hyperparameters Mean"]
    hyperparams_cov[idx, ...] = np.array(
        genList[i]["Solver"]["Current Hyperparameters Covariance"])
    gammas[idx] = genList[i]["Solver"]["Gamma"]
    total_logprobab[idx] = genList[i]["Solver"][
        "Current Sample Log Probability"]
    total_logprior[idx] = genList[i]["Solver"]["Current Sample Log Prior"]
    total_logllh[idx] = genList[i]["Solver"]["Current Sample Log Likelihood"]
    if genList[i]["Solver"]["Log All Samples"]:
      log_all_samples = True

  total_logprior_check[:] = np.array(
      [tot - l for tot, l in zip(total_logprobab, total_logllh)])
  assert np.all(
      np.isclose(
          # where one of the two contains an inf, the prior might still be finite, so we reduce to finite values
          total_logprior_check[np.isfinite(total_logprobab)],
          total_logprior[np.isfinite(total_logprobab)]))

  # Get all samples across sampling steps, whenever available
  if log_all_samples:
    all_samples = [[] for _ in range(numChains)]
    all_samples_x = [[] for _ in range(numChains)]
    all_llhs = [[] for _ in range(numChains)]
    all_priors = [[] for _ in range(numChains)]
    for idx, i in enumerate(genList):
      if genList[i]["Solver"]["Log All Samples"]:
        for c in range(numChains):
          # chain_samples = np.array(genList[i]["Solver"][ "All Samples This Iteration" ][c]))  # 0 (step) -> 1 (indiv) and 1 -> 0
          all_samples[c].extend(
              np.array(genList[i]["Solver"]["All Samples This Iteration"][c]))
          all_samples_x[c].extend(
              list(np.arange(i, i + 1, 1. / (nOuter * nInner))))
          all_llhs[c].extend(
              genList[i]["Solver"]["All Loglikelihoods This Iteration"][c])
          all_priors[c].extend(
              genList[i]["Solver"]["All Priors This Iteration"][c])
      else:
        for c in range(numChains):
          all_samples[c].extend([genList[i]["Solver"]["Current Samples"][c]])
          all_samples_x[c].extend([float(i)])
          all_llhs[c].extend(
              [genList[i]["Solver"]["Current Sample Log Likelihoods"][c]])
          all_priors[c].extend(
              [genList[i]["Solver"]["Current Sample Log Priors"][c]])

  # Create an "outer" subplot for the two column titles
  fig, ax0 = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 20), dpi=60)
  ax0[0].set_title(
      "Latent Variables", y=1.03, fontdict={
          'fontweight': 8,
          'fontsize': 20
      })
  ax0[1].set_title(
      "Hyperparameters: Mean",
      y=1.03,
      fontdict={
          'fontweight': 8,
          'fontsize': 20
      })
  ax0[2].set_title(
      "Hyperparameters: Cov \nGamma \nLog-Probabilities",
      y=1.03,
      fontdict={
          'fontweight': 8,
          'fontsize': 20
      })
  #ax0[1].set_title("Latent Variables, All Samples", y=1.03, fontdict={'fontweight': 8, 'fontsize': 20})
  # ax0[3].set_title("Log-Probability", y=1.03, fontdict={'fontweight': 8, 'fontsize': 20})

  # Make outer plot invisible
  for i in range(ncols):
    ax0[i].tick_params(
        labelcolor=(1., 1., 1., 0.0),
        top='off',
        bottom='off',
        left='off',
        right='off')
    ax0[i]._frameon = False
    ax0[i].tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

  fig.subplots_adjust(hspace=.8)

  # suptitle?
  plt.suptitle(
      'SAEM Plotter \nNumber of Chains: {0}\n'.format(str(numChains)),
      fontweight='bold',
      fontsize=20)
  # 1st column: Sample means in generation steps
  for dim in range(numLatentSpaceDims):
    ax = fig.add_subplot(nrows, ncols, ncols * dim + 1)
    for indiv in range(numIndividuals):
      ax.errorbar(
          x=generations,
          y=sample_means[:, indiv, dim],
          yerr=sample_sdevs[:, indiv, dim],
          c=colors(indiv))
    ax.set_title("All individuals, " + latentVars[dim]["Name"])
    plt.xlabel("Generation")

  # fig, ax = plt.subplots(nrows=max(numLatent, numHyper) + 1, ncols=1, figsize=(15,8))
  # 2nd column: The "mean"-hyperparameters
  for dim in range(numLatentSpaceDims):
    ax = fig.add_subplot(nrows, ncols, ncols * dim + 2)
    ax.plot(generations, hyperparams_mean[:, dim])
    ax.set_title("Mean for transformed " + latentVars[dim]["Name"])
    plt.xlabel("Generation")

  # 3rd column: Eigenvalues of the covariance-matrix-hyperparameter
  assert hyperparams_cov.shape[0] == numgens
  all_eigvals = np.zeros((numgens, numLatentSpaceDims))
  for st in range(numgens):
    cov = hyperparams_cov[st]
    eigvals = np.linalg.eigvals(cov)
    all_eigvals[st] = eigvals
  # sort eigenvalues; try to plot like a heatmap
  all_eigvals = np.array(
      sorted(all_eigvals.transpose([1, 0]),
             key=lambda seq: seq[-1])).transpose([1, 0])
  eigcols = get_cmap(numLatentSpaceDims, 'copper')  # 'hot', 'summer', 'afmhot'
  ax = fig.add_subplot(nrows, ncols, ncols * 0 + 3)
  for dim in range(numLatentSpaceDims):
    ax.plot(generations, all_eigvals[:, -dim - 1], c=eigcols(dim))
  #plt.yscale('log')
  plt.xlabel("Generation")
  plt.ylabel("Eigenvalues")
  ax.set_title("Eigenvalues of the covariance matrix")

  # Gamma: (Plot them below the eigenvalues, in 2nd row
  ax = fig.add_subplot(nrows, ncols, ncols * 1 + 3)
  ax.plot(generations, gammas)
  ax.set_title("Gamma (Robbins-Monro decay)")
  plt.xlabel("Generation")

  # Log-probabilities across generations
  ax = fig.add_subplot(nrows, ncols, ncols * 2 + 3)
  ax.plot(generations, total_logprobab, c='k')
  ax.plot(generations, total_logllh, c='orange')
  ax.plot(generations, total_logprior, c='yellow')
  plt.xlabel("Generation")
  ax.set_title(
      "Black: Total log-probability \n Orange: Total log-likelihood \n Yellow: Total log-prior \n (Mean over chains)"
  )
  #plt.title("Log-Probability", y=1.03, fontdict={'fontweight': 8, 'fontsize': 20})
  # Additional figure: All samples
  numrows2 = numLatentSpaceDims + 3
  fig2, ax2 = plt.subplots(nrows=numrows2, ncols=1, figsize=(20, 26))
  plt.suptitle("SAEM Plotter, Page 2:\n"
               "Latent Variables, All Samples\n",
               fontweight='bold',
               fontsize=20)
  #ax2.set_title("Latent Variables, All Samples", y=1.03, fontdict={'fontweight': 8, 'fontsize': 20})

  # fig2.suptitle("Hyperparameters: Eigenvalues of Covariance", y=1.03, fontdict={'fontweight': 8, 'fontsize': 20})
  fig2.subplots_adjust(hspace=.8)
  for ax in ax2:
    ax.tick_params(
        labelcolor=(1., 1., 1., 0.0),
        top='off',
        bottom='off',
        left='off',
        right='off')
    ax._frameon = False
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

  # Sort, before plotting
  for c in range(numChains):
    A, L, P, X = zip(*(sorted(
        zip(all_samples[c], all_llhs[c], all_priors[c], all_samples_x[c]),
        key=lambda tup: tup[-1])))
    all_samples[c], all_llhs[c], all_priors[c], all_samples_x[c] = np.array(
        A), np.array(L), np.array(P), np.array(X)
  for dim in range(numLatentSpaceDims):
    ax = fig2.add_subplot(numrows2, 1, dim + 1)
    for indiv in range(numIndividuals):
      for c in range(numChains):
        ax.plot(
            all_samples_x[c],
            all_samples[c][:, indiv, dim],
            c=colors(indiv),
            alpha=0.6,
            lw=0.2)
        # ax.errorbar(x=generations, y=sample_means[:, indiv, dim], yerr=sample_sdevs[:, dim], c=colors[indiv])
    ax.set_title("All individuals, " + latentVars[dim]["Name"])
    # plt.xlabel("Generation")
    plt.ylabel("Sampled value")

  # add one plot for each of: loglikelihood, logprior, their sum
  ax = fig2.add_subplot(numrows2, 1, numLatentSpaceDims + 1)
  ax.set_title("Total log-probability per chain and individual")
  for indiv in range(numIndividuals):
    for c in range(numChains):
      ax.plot(
          all_samples_x[c], [
              l + p
              for l, p in zip(all_llhs[c][:, indiv], all_priors[c][:, indiv])
          ],
          c=colors(indiv),
          alpha=0.6,
          lw=0.3)
  logp_mean = np.mean(np.array(all_llhs) + np.array(all_priors), axis=(0, 2))
  ax.plot(all_samples_x[0], logp_mean, c='k', alpha=1, lw=0.4)
  # plt.xlabel("Generation")
  plt.ylabel("Log-P")

  ax = fig2.add_subplot(numrows2, 1, numLatentSpaceDims + 2)
  ax.set_title("Log-likelihood per chain and individual")
  for indiv in range(numIndividuals):
    for c in range(numChains):
      ax.plot(
          all_samples_x[c],
          all_llhs[c][:, indiv],
          c=colors(indiv),
          alpha=0.6,
          lw=0.3)
  llh_mean = np.mean(np.array(all_llhs), axis=(0, 2))
  ax.plot(all_samples_x[0], llh_mean, c='k', alpha=1, lw=0.4)
  #plt.xlabel("Generation")
  plt.ylabel("Log-LLH")

  ax = fig2.add_subplot(numrows2, 1, numLatentSpaceDims + 3)
  ax.set_title("Log-prior per chain and individual")
  for indiv in range(numIndividuals):
    for c in range(numChains):
      ax.plot(
          all_samples_x[c],
          all_priors[c][:, indiv],
          c=colors(indiv),
          alpha=0.6,
          lw=0.3)
  prior_mean = np.mean(np.array(all_priors), axis=(0, 2))
  ax.plot(all_samples_x[0], prior_mean, c='k', alpha=1, lw=0.4)
  plt.xlabel(
      "Generation\nNote: Prior is set to 0 for sub-steps 1 to N1 \n\n(Black lines: mean over all individuals and chains)"
  )
  plt.ylabel("Log-Prior")

  return (fig, fig2), ax


def get_cmap(n, name='hsv'):
  '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    Credits to Ali @ https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    '''
  return plt.cm.get_cmap(name, n)


def plot(genList, args):

  fig, ax = plotVariables(genList)
