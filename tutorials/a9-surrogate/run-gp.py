#!/usr/bin/env python3

def read_matrix_for_gp( fileName, lastColumnIsData=True ):
  f = open( fileName, "r" )
  data = [ [ float(n) for n in line.split()]  for line in f]
  f.close()

  if( lastColumnIsData == True ):
    x = [row[:-1] for row in data]
    y = [row[ -1] for row in data]
  else:
    x = data
    y = []

  return x,y


import os
import sys
import korali

k = korali.initialize()

k["Problem"]["Type"] = "Evaluation/GaussianProcess"
k["Problem"]["Covariance Function"] = "CovSum ( CovSEiso, CovNoise)"
x, y = read_matrix_for_gp('data/sincos3d_train.dat')
k["Problem"]["X Data"] = x
k["Problem"]["Y Data"] = y

k["Solver"]["Type"] = "Optimizer/Rprop"
k["Solver"]["Termination Criteria"]["Max Generations"] = 200
k["Solver"]["Termination Criteria"]["Parameter Relative Tolerance"] = 1e-8

k["Console Output"]["Verbosity"] = "Normal"
k["Results Output"]["Path"] = "_korali_result_train"
k.run()


k = korali.initialize()
k["Problem"]["Type"] = "Execution/GaussianProcess"
k["Problem"]["Gaussian Process Json File"] =  "_korali_result_train/final.json"
x, y = read_matrix_for_gp('data/sincos3d_test.dat')
k["Problem"]["X Data"] = x
k["Problem"]["Y Data"] = y

k["Solver"]["Type"] = "Executor"
k["Solver"]["Executions Per Generation"] = 1

k["Console Output"]["Verbosity"] = "Normal"
k["Results Output"]["Path"] = "_korali_result_test"
k.run()


k = korali.initialize()
k["Problem"]["Type"] = "Execution/GaussianProcess"
k["Problem"]["Gaussian Process Json File"] =  "_korali_result_train/final.json"
x, y = read_matrix_for_gp('data/sincos3d_new.dat',lastColumnIsData=True)
k["Problem"]["X Data"] = x
k["Problem"]["Y Data"] = y

k["Solver"]["Type"] = "Executor"
k["Solver"]["Executions Per Generation"] = 1

k["Console Output"]["Verbosity"] = "Normal"
k["Results Output"]["Path"] = "_korali_result_new"
k.run()
