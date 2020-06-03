import numpy as np

  def verifyMean(samples, expected, atol = 0.01) :

                 d = len(expected)
                           samples = np.reshape(samples, (- 1, d)) mu = np.mean(samples, axis = 0)

                                                                                  for i in range(d) : assert np.isclose(expected[i], mu[i], atol = atol), "Sample mean deviates from analytical Mean ({0} vs {1})".format(expected, mu)

                                                                                                                                                                                                                            def verifyStd(samples, expected, atol = 0.1) :

                                                                                                                                                                                                                                          d = len(expected) samples = np.reshape(samples, (- 1, d)) std = np.std(samples, axis = 0)

                                                                                                                                                                                                                                                                                                                   for i in range(d) : assert np.isclose(expected[i], std[i], atol = atol), "Sample std deviates from analytical Std ({0} vs {1})".format(expected, std)
