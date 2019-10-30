import numpy as np

def verifyMean(k, expected):
    samples = k["Solver"]["Internal"]["Sample Database"]
    
    mu = np.mean(samples)
    print(mu)
    
    assert np.isclose(expected, mu), "Sample mean deviates from analytical Mean"

def verifyStd(k, expected):
    samples = k["Solver"]["Internal"]["Sample Database"]

    std     = np.std(samples)
    print(std)
    assert np.isclose(expected, std), "Sample std deviates from analytical Std"

 
