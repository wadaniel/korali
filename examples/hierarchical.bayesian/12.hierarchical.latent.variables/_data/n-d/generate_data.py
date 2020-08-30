import numpy as np
import sys
sys.path.append('./_model')
sys.path.append('../../_model')
from utils import *

#
#  Model 4:
#    draw z_i ~ N(theta, Omega), where Omega need not be diagonal
#       psi_i = f(z_i), where f is either id, log^-1, or logit^-1
#    draw x_i ~ N(psi_i, sigma**2)
#


def draw_from_hierarchical(n_individuals, sigma, cov, mean, max_n_samples,
                           d_normal, d_logn, d_logitn):
  '''
        x_ij ~ N(latent_i, sigma**2)
        latent_i / log(latent_i) / logit(latent_i) ~ N(mean, cov)
    '''
  results = []
  d = d_normal + d_logn + d_logitn
  latents = np.zeros(shape=(n_individuals, d))
  latents_z = np.zeros(shape=(n_individuals, d))
  for i in range(n_individuals):
    n_samples = np.random.choice(np.arange(1, max_n_samples + 1))

    # a) sample the latent variable
    latents_z[i, :] = np.random.multivariate_normal(mean, cov)
    tot_dim = 0
    for dim in range(d_normal):
      latents[i, tot_dim] = latents_z[i, tot_dim]
      tot_dim += 1
    for dim in range(d_logn):
      latents[i, tot_dim] = np.exp(latents_z[i, tot_dim])
      tot_dim += 1
    for dim in range(d_logitn):
      latents[i, tot_dim] = inv_logit(latents_z[i, tot_dim])
      tot_dim += 1

    # b) sample each sample
    indiv_results = []
    for j in range(n_samples):
      # sample data point
      pt = np.random.multivariate_normal(latents[i], np.eye(d) * sigma**2)
      indiv_results.append(pt)
    results.append(indiv_results)
  return results, latents, latents_z


def generate_data_advanced():
  '''
        multiple dimensions and multiple data points per individual possible
        given latent variables, the sampled points simply are normally distributed, with sdev sigma, around the latent variable
        '''
  n_individuals = 5
  max_n_samples = 10  # each individual has between 1 and this number of data points assigned
  sigma = .2  # Note: for logit-normal variables, a sigma of 0.5 is already pretty high.
  omega1 = 1.0
  omega2 = 0.25
  omega3 = 0.5
  d_normal = 1
  d_logn = 2
  d_logitn = 2
  d_latent = d_normal + d_logn + d_logitn
  d = d_latent
  # The hyperparameter
  mean = np.arange(d_latent)
  #mean = [-2, 0 , 1, 2]
  Omega = np.eye(d_latent)  # the covariance matrix
  Omega *= omega1
  Omega[0, 1] = omega2
  Omega[1, 0] = omega2  # the first two latent variables are correlated
  Omega[-1, -1] = omega3  # the last variable has a different variance

  output_file = "data_advanced.in"
  info_output_file = "data_advanced_info.txt"

  data, latents, latents_z = draw_from_hierarchical(
      n_individuals,
      sigma=sigma,
      cov=Omega,
      mean=mean,
      max_n_samples=max_n_samples,
      d_normal=d_normal,
      d_logn=d_logn,
      d_logitn=d_logitn)

  # store to file:
  with open(output_file, "w") as fd:
    fd.write(
        str(n_individuals) + " " + str(d) + " " + str(max_n_samples) + " " +
        str(sigma) + " " + str(d_normal) + " " + str(d_logn) + " " +
        str(d_logitn) + "\n")
    for i in range(n_individuals):
      n_samples = len(data[i])
      fd.write("N " + str(n_samples) +
               "\n")  # how many points there are for this individual
      lines = [
          " ".join([str(data[i][j][k])
                    for k in range(d_latent)]) + "\n"
          for j in range(n_samples)
      ]
      fd.writelines(lines)
    fd.write("# Header info: \n")
    fd.write(
        "# nr individuals | data dimension | max nr samples each | sigma | d_normal | d_lognormal | d_logitnormal "
    )

  # store ground truth to another file:
  with open(info_output_file, "w") as fd:
    fd.write("### This file contains the generation information for " +
             output_file + " ###\n")
    fd.write(
        "#############################################################################\n\n"
    )
    fd.write("### 1. True hyperparameters and latent variables: \n")
    fd.write("n_individuals: " + str(n_individuals) + "\n")
    fd.write("d, data dimensions: " + str(d) + "\n")
    fd.write("sigma: " + str(sigma) + "\n")
    fd.write("d_normal: " + str(d_normal) + "| " + "d_logn: " + str(d_logn) +
             "| " + "d_logitn: " + str(d_logitn) + "\n")
    fd.write("Mean (around which transformed latents are scattered): \n")
    fd.write("\t " + str(mean) + "\n")
    fd.write("Omega for latent_i ~ N(mean, Omega): \n")
    fd.write("\t[\n")
    for i in range(d_latent):
      fd.write("\t [" + ", ".join(Omega[i, :].astype(str)) + "],\n")
    fd.write("\t]\n")
    fd.write(
        "---------------------------------------------------------------\n")
    # latent variables
    fd.write("All %d latent variable vectors:\n" % n_individuals)
    for i in range(n_individuals):
      fd.write("\t[" + ", ".join(latents[i].astype(str)) + "]\n")

    fd.write(
        "---------------------------------------------------------------\n")
    # latent variables
    fd.write("All %d transformed latent variable vectors (i.e. z):\n" %
             n_individuals)
    for i in range(n_individuals):
      fd.write("\t[" + ", ".join(latents_z[i].astype(str)) + "]\n")

    fd.write(
        "---------------------------------------------------------------\n\n")
    fd.write("### 2. Heuristic optimizers given only the data: \n")
    # More statistics to allow easier interpretation of the results
    fd.write(
        "\nBest latent z-variable estimate per individual (completely ignoring shared prior):\n"
    )
    individual_z_mean = []
    for i in range(n_individuals):
      points_i = data[i]
      z_i = [transform_to_z(pt, d_normal, d_logn, d_logitn) for pt in points_i]
      z_mean = np.mean(np.array(z_i), axis=0)
      individual_z_mean.append(z_mean)
      fd.write("Indiv. " + str(i) + ":\t[" + ", ".join(z_mean.astype(str)) +
               "]\n")

    fd.write(
        "\nBest hyperparameter-mean estimate (completely ignoring that there are different individuals):\n"
    )
    mean_ = []  #np.zeros(shape=(d_normal + d_logn + d_logitn))
    for i in range(n_individuals):
      points_i = data[i]
      z_i = [transform_to_z(pt, d_normal, d_logn, d_logitn) for pt in points_i]
      for z_i_ in z_i:
        mean_.append((z_i_))
    mean_ = np.mean(mean_, axis=0)
    fd.write("\t[" + ", ".join(mean_.astype(str)) + "]\n")

    fd.write("\nMean of individuals optimal z:\n")
    individual_z_mean_mean = np.mean(individual_z_mean, axis=0)
    fd.write("\t[" + ", ".join(individual_z_mean_mean.astype(str)) + "]\n")

    fd.write(
        "\n'Sample' covariance, using the individuals' data means as samples:\n"
    )
    sample_cov = np.cov(
        np.array(individual_z_mean).transpose(),
        bias=True)  # (H)SAEM also uses a biased estimate.
    fd.write("\t[\n")
    for i in range(d_latent):
      fd.write("\t [" + ", ".join(sample_cov[i, :].astype(str)) + "],\n")
    fd.write("\t]\n")
    fd.write(
        "---------------------------------------------------------------\n")

    # Total log-probability of the true data with true hyperparams and true latent variables
    try:
      from scipy.stats import multivariate_normal
      do_calc_probab = True
    except ImportError as e:
      print(
          "Scipy not installed, skipping probability calcuation for generated data."
      )
      do_calc_probab = False

    if do_calc_probab:
      fd.write(
          "### 3. Log-probabilities of the original data (any optimizer should do at least as well): \n"
      )
      logp_latent = []
      logp_points_conditional = []
      mvn = multivariate_normal(mean=mean, cov=Omega)
      for z in latents_z:
        logp_latent.append(mvn.logpdf(z))
      for i in range(n_individuals):
        mvn = multivariate_normal(
            mean=latents[i], cov=sigma * np.eye(len(latents[i])))
        logp_points_conditional.append(0)
        for pt in data[i]:
          logp_points_conditional[i] += mvn.logpdf(pt)
      total_log_prob = np.sum(logp_points_conditional) + np.sum(
          logp_points_conditional)
      individual_log_prob = [
          logp_points_conditional[i] + logp_latent[i]
          for i in range(n_individuals)
      ]
      fd.write("\nTotal log-probability: %.3f\n" % total_log_prob)
      fd.write("\nIndividual log-probability:\n")
      for i in range(n_individuals):
        fd.write("\tIndiv. %d: \t %.2f\n" % (i, individual_log_prob[i]))

      fd.write("\nIndividual log-llh:\n")
      for i in range(n_individuals):
        fd.write("\tIndiv. %d: \t %.2f\n" % (i, logp_points_conditional[i]))

      fd.write("\nIndividual log-prior:\n")
      for i in range(n_individuals):
        fd.write("\tIndiv. %d: \t %.2f\n" % (i, logp_latent[i]))

  print("Done generating n-d data.")


if __name__ == '__main__':
  generate_data_advanced()
