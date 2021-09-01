#!/usr/bin/env python
import numpy 
import korali
from mpi4py import MPI

# This toy model has no other purpose rather than showing how to use MPI4py in Korali
# It evaluates different x (one per worker) and returns its average as team evaluation.
def model(p):
  comm = korali.getWorkerMPIComm()
  rank = comm.Get_rank()
  size = comm.Get_size()
  
  x = p["Parameters"][0]
  y = -0.5 * x * x
  
  # Testing reduce operation
  recvdata = numpy.zeros(size,dtype=numpy.float)
  senddata = numpy.arange(size,dtype=numpy.float)
  senddata[0] = y
  comm.Allreduce(senddata, recvdata, op=MPI.SUM)
  
  y = recvdata[0]
  if (rank == 0): print(f"MPI Rank: {rank}/{size} - Evaluation: f({x}) = {y}")
  p["F(x)"] = y
  

