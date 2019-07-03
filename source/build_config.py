#!/usr/bin/env python3

import os

# Finding Solver Header Files

path = '../include/solvers'

files = []
for r, d, f in os.walk(path):
 for file in f:
   if '.h' in file:
    files.append(os.path.join(r, file))

# Parse each file
for f in files:
 print(f)
 
 # Starting from initial state
 step = 0
 
 with open(f, 'r') as file_object:
  line = file_object.readline()
  while line:
   
   if (step == 0):
    print(line)
    step = step + 1
    
   line = file_object.readline()