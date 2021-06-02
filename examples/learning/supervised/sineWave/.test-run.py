#! /usr/bin/env python3
from subprocess import call
import os

os.environ["OMP_NUM_THREADS"] = "4"
  
r = call(["python3", "run-ffn.py", "--optimizer", "AdaBelief", "--learningRate", "0.005"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer", "Adagrad", "--learningRate", "0.05"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer", "Adam", "--learningRate", "0.005"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer", "MADGRAD", "--learningRate", "0.005"])
if r!=0:
  exit(r)
      
r = call(["python3", "run-ffn.py", "--optimizer", "RMSProp", "--learningRate", "0.002"])
if r!=0:
  exit(r)
    
r = call(["python3", "run-rnn.py", "--optimizer", "AdaBelief", "--learningRate", "0.001"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--optimizer", "Adagrad", "--learningRate", "0.01"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--optimizer", "Adam", "--learningRate", "0.001"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--optimizer", "MADGRAD", "--learningRate", "0.001"])
if r!=0:
  exit(r)
      
r = call(["python3", "run-rnn.py", "--optimizer", "RMSProp", "--learningRate", "0.001"])
if r!=0:
  exit(r)
  
exit(0)
