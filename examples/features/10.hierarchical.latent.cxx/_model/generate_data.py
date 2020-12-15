import numpy as np

#
#  Model 3:
#    draw psi_i ~ N(theta, omega**2)
#    draw x_i ~ N(psi_i, sigma**2)
#


def draw_from_simple_hierarchical(n_individuals,
                                  sigma,
                                  omega,
                                  mean,
                                  n_samples=1):
  ''' '''
  results = np.zeros((n_individuals, n_samples))
  latent_vars = np.random.normal(mean, scale=omega, size=(n_individuals,))
  for i in range(n_individuals):
    lvar = latent_vars[i]
    results[i, :] = np.random.normal(lvar, scale=sigma, size=(n_samples,))
  return results


def inv_logit(z):
  return 1. / (1. + np.exp(-z))


def logit(p):
  assert 0 <= p < 1
  return np.log(p / (1. - p))


def transform_to_z(point, d_normal, d_lognormal, d_logitnormal):
  ''' Assumes coordinates in :param point are ordered: normal, then lognormal, then logitnormal.'''
  z = np.zeros_like(point)
  eps = 1.e-15  # cannot be smaller than around 1.e-17
  d = d_normal + d_lognormal + d_logitnormal
  z[:d_normal] = point[:d_normal]
  z[d_normal:d_normal + d_lognormal] = [
      (np.log(p) if p > 0 else np.log(eps))
      for p in point[d_normal:d_normal + d_lognormal]
  ]
  z[d - d_logitnormal:] = [(logit(p) if (0 <= p < 1) else
                            (logit(eps) if (p < 0) else logit(1 - eps)))
                           for p in point[d - d_logitnormal:]]
  return z


def transform_from_z(z, d_normal, d_lognormal, d_logitnormal):
  point = np.zeros_like(z)
  d = d_normal + d_lognormal + d_logitnormal
  point[:d_normal] = z[:d_normal]
  point[d_normal:d_normal +
        d_lognormal] = [np.exp(p) for p in z[d_normal:d_normal + d_lognormal]]
  point[d - d_logitnormal:] = [inv_logit(p) for p in z[d - d_logitnormal:]]
  return point


def test_transform_to_z():
  for i in range(30):
    d_normal, d_lognormal, d_logitnormal = np.random.choice(
        list(np.arange(10)), 3)
    d = d_normal + d_lognormal + d_logitnormal
    z = np.random.uniform(-10, 10, (d,))
    point_0 = z[:d_normal].tolist()
    point_1 = [np.exp(z_) for z_ in z[d_normal:d_normal + d_lognormal]]
    point_2 = [inv_logit(z_) for z_ in z[d - d_logitnormal:]]
    point = np.array(point_0 + point_1 + point_2)
    if not np.allclose(
        z,
        transform_to_z(point, d_normal, d_lognormal, d_logitnormal),
        rtol=1.e-8):
      print("aaaah")
    assert np.allclose(
        z,
        transform_to_z(point, d_normal, d_lognormal, d_logitnormal),
        rtol=1.e-8)


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


# def draw_from_multiple_gaussian(nr_points=10, dimensions=2, mus=[[0, 0],[2,2]], sigma=1):
#     results = np.zeros((nr_points, dimensions))
#     cluster_assignment = np.zeros((nr_points,), dtype=int)
#     nr_clusters = len(mus)
#     for i in range(nr_points):
#         c = np.random.choice(list(range(nr_clusters)))
#         mu = mus[c]
#         pt = np.random.normal(loc=mu, scale=sigma)
#         cluster_assignment[i] = c
#         results[i, :] = pt
#     return results, cluster_assignment


def generate_data():
  n_individuals = 10
  n_samples = 1  # how many datapoints per individual
  sigma = .5
  omega = 2.
  mean = 1  # The Hyperparameter
  output_file_925 = "data_925.in"  # 925 for section 9.2.5
  info_output_file_925 = "data_925_info.txt"

  simple_data = draw_from_simple_hierarchical(
      n_individuals, sigma=sigma, omega=omega, mean=mean, n_samples=n_samples)

  # store to files:
  with open(output_file_925, "w") as fd:
    d = simple_data
    fd.write(
        str(n_individuals) + " " + str(n_samples) + " " + str(sigma) + " " +
        str(omega) + "\n")
    lines = [
        " ".join([str(d[i, j])
                  for j in range(n_samples)]) + "\n"
        for i in range(n_individuals)
    ]
    fd.writelines(lines)

  # calculate the optimizer (just the mean) and store it as reference
  with open(info_output_file_925, "w") as fd:
    fd.write("# The optimizing hyperparameter (the mean):" + "\n")
    optimizer = np.mean(simple_data)
    fd.write("# \t%.3f\n" % optimizer)
    fd.write("# Original mean (used to create data):\n")
    fd.write("# \t" + str(mean) + "\n")
    fd.write("# Original hyperparameter variance:\n# \tw^2 with w == " +
             str(omega) + "\n")
  print(
      "Done generating data according to section 9.2.5 in 'Mixed Effect Models' by M. Lavielle."
  )


def generate_data_advanced():
  '''
        multiple dimensions and multiple data points per individual possible
        given latent variables, the sampled points simply are normally distributed, with sdev sigma, around the latent variable
        '''
  n_individuals = 200
  max_n_samples = 10  # each individual has between 1 and this number of data points assigned
  sigma = .4  # Note: for logit-normal variables, a sigma of 0.5 is already pretty high.
  omega1 = 1.0
  omega2 = 0.25
  omega3 = 0.5
  d_normal = 2
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

  print("Done generating more complicated data")


if __name__ == '__main__':
  generate_data()
  generate_data_advanced()
