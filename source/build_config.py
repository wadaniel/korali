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
 print('Processing: ' + f + '...')
 
 # Starting from initial state
 step = 0
 
 with open(f, 'r') as file_object:
  line = file_object.readline()
  while line:
   
   ## Step 0: Look for Solver Name
   if (step == 0):
    if (line.startswith('Solver Name:')):
     solverName = line.replace('Solver Name:', '').strip()
     print(solverName)
     step = step + 1
   
   ## Step 1: Look for Solver Type
   if (step == 1):
    if (line.startswith('Type:')):
     solverType = line.replace('Type:', '').strip()
     print(solverType)
     step = step + 1
   
   ## Step 2: Look for Solver Description  
   if (step == 2):
    if (line.startswith('Description:')):
     line = file_object.readline()
     solverDescription = ''
     step = step + 1
    
   ## Step 3: Fill-in Solver Description
   if (step == 3):
    if (line.startswith('*******************************')):
     print(solverDescription)
     step = step + 1
    else:
     solverDescription += line

   line = file_object.readline()