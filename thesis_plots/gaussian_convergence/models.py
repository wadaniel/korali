import numpy as np

def lLaplaceModel(s, mu, sigma):
    x0 = s["Parameters"][0]
    scale = 1.0/np.sqrt(2.0) * sigma
    r = -np.log(2.0 * scale) - abs(x0 - mu) / scale
    s["logP(x)"] = r
    s["grad(logP(x))"] = [-np.sign(x0 - mu) / scale]
    s["H(logP(x))"] = [[0.0]]

def laplaceDensity(x, mu, sigma):
    scale = 1.0/np.sqrt(2.0) * sigma
    return 1.0 / (2*scale) * np.exp(-np.abs(x-mu)/scale)

def lGaussianModel(s, mu, sigma):
    v = s["Parameters"][0]
    s["logP(x)"] = -0.5 * (v - mu)**2/sigma**2
    s["grad(logP(x))"] = [-(v - mu)/sigma**2]
    s["H(logP(x))"] = [[-1.0/sigma**2]]

def gaussianDensity(x, mu, sigma):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x-mu)/sigma)**2)
