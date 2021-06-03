#! /usr/bin/env python3
from subprocess import call
import os

r = call(["python3", "run-ffn.py", "--optimizer", "AdaBelief", "--learningRate", "0.005", "--maxGenerations", "10"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer", "Adagrad", "--learningRate", "0.05", "--maxGenerations", "10"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer", "Adam", "--learningRate", "0.005", "--maxGenerations", "10"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer", "MADGRAD", "--learningRate", "0.005", "--maxGenerations", "10"])
if r!=0:
  exit(r)
      
r = call(["python3", "run-ffn.py", "--optimizer", "RMSProp", "--learningRate", "0.002", "--maxGenerations", "10"])
if r!=0:
  exit(r)
  
r = call(["python3", "run-ffn.py", "--optimizer", "Adam", "--learningRate", "0.005", "--engine", "Korali", "--maxGenerations", "10"])
if r!=0:
  exit(r)
    
r = call(["python3", "run-rnn.py", "--optimizer", "AdaBelief", "--learningRate", "0.001", "--maxGenerations", "10"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--optimizer", "Adagrad", "--learningRate", "0.01", "--maxGenerations", "10"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--optimizer", "Adam", "--learningRate", "0.001", "--maxGenerations", "10"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--optimizer", "MADGRAD", "--learningRate", "0.001", "--maxGenerations", "10"])
if r!=0:
  exit(r)
      
r = call(["python3", "run-rnn.py", "--optimizer", "RMSProp", "--learningRate", "0.001", "--maxGenerations", "10"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--optimizer", "Adam", "--learningRate", "0.001", "--rnnType", "LSTM", "--maxGenerations", "10"])
if r!=0:
  exit(r)
    
exit(0)
