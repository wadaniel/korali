#!/usr/bin/env python3
import numpy as np

T = 5
M = 10000

S0 = 50
Q0 = 1e4
sigma = .95
lam = 1e-6
eta = 2.5e-4
gamma = 2.5e-5

maxSteps = 100
dt = T/maxSteps
shift = 0. #900000
eps = 0.01


def transform(q, t, q0=Q0, t0=T):
    assert q<=q0, print(q)
    assert t<=t0, print(t)

    return q/q0, t/T
    # TODO: sth wrong with transform
    qhat = q/q0-1
    that = t/T
    r = np.sqrt(q**2+t**2)
    th = np.arctan(-t/q)
    eta = -that/qhat

    if th <= np.pi/4:
        rtilde = r*np.sqrt((eta**2+1)*(2.*np.cos(np.pi/4-th)**2))
    else:
        rtilde = r*np.sqrt((eta**(-2)+1)*(2.*np.cos(th-np.pi/4)**2))


    qtilde = -rtilde*np.cos(th)
    xtilde = rtilde*np.sin(th)

    print(qtilde)
    print(xtilde)

    assert qtilde >= -1
    assert xtilde >= -1

    assert qtilde <= 1
    assert xtilde <= 1

    return qtilde, xtilde



def env(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]

 Q = Q0
 t = 0.

 X = np.zeros(M)
 S = S0*np.ones(M)

 qtilde, ttilde = transform(Q,t)
 s["State"] = [qtilde, ttilde]

 rOld = 0.
 r = 0.

 cr = 0.
 while t < T-1e-6:

  # Getting new action
  s.update()
  
  # Performing the action
  f = s["Action"][0]

  n = Q * f 
  t += dt
  
  # Terminal sell
  if t >= T-1e-12:
    n = Q

  Q = Q - n
  Q = np.clip(Q, a_min=0, a_max=np.inf)

  P = S-gamma*n
  X += P*n
 
  S = S + sigma*np.sqrt(dt)*np.random.normal(loc=0., scale=1., size=M) - eta*n
  S = np.clip(S, a_min=eps, a_max=np.inf)
 
  IS = Q0*S0 - X

  rOld = r
  r = np.mean(IS) + lam * np.var(IS) - shift

  # Getting Reward
  s["Reward"] = -r+rOld
   
  # Storing New State
  qtilde, ttilde = transform(Q,t)
  s["State"] = [qtilde, ttilde] 
  cr += (-r+rOld)

 # Setting finalization status
 s["Termination"] = "Terminal"
