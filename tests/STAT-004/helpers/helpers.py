import numpy as np

def verifyMean(k, expected, atol=0.01):
    samples = k["Solver"]["Internal"]["Sample Database"]
    
    d = len(expected)
    samples = np.reshape(samples,(-1,d))
    mu = np.mean(samples, axis=0)

    for i in range(d):
        assert np.isclose(expected[i], mu[i], atol = atol), "Sample mean deviates from analytical Mean ({0} vs {1})".format(expected, mu)

def verifyStd(k, expected, atol = 0.1):
    samples = k["Solver"]["Internal"]["Sample Database"]

    std     = np.std(samples)
    assert np.isclose(expected, std, atol = atol), "Sample std deviates from analytical Std ({0} vs {1})".format(expected, std)

 
