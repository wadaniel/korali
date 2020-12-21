#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil

def model(x, resultFolder, objective, tmax):
 
 SourceFolderName = "_config"
 DestinationFolderName = resultFolder + '/sample' + str(x["Sample Id"]).zfill(6) 
 
 # Copy the 'model' folder into a temporary directory
 if os.path.exists( DestinationFolderName ):
  shutil.rmtree( DestinationFolderName)
 shutil.copytree( SourceFolderName, DestinationFolderName )

 CurrentDirectory = os.getcwd()

 # Move inside the temporary directory
 try:
  os.chdir( DestinationFolderName )
 except OSError as e:
  print("I/O error(" + str(e.errno) + "): " + e.strerror )
  print("The folder " + DestinationFolderName + " is missing")
  sys.exit(1)
 
 # Storing base parameter file
 configFile='par.py'
 with open(configFile, 'a') as f:
  f.write('bone_factor = %.10f\n' % x["Parameters"][0] )
  
 # Run Aphros for this sample
 sampleOutFile = open('sample.out', 'w')
 sampleErrFile = open('sample.err', 'w')
 subprocess.call(['bash', 'run.sh'], stdout=sampleOutFile, stderr=sampleErrFile)

 # Loading results from file
 resultFile = 'stat.dat'
 try:
  with open(resultFile) as f:
   resultLines = f.readlines()
 except IOError:
  print("[Korali] Error: Could not load result file: " + resultFile)
  exit(1)
 
 # Obtaining result column names
 columnNames = resultLines[0].split()
 
 # Trying to find objective column
 try:
  objectiveCol = columnNames.index(objective)
 except ValueError:
  print("[Korali] Error: Could not load objective column name: '" + objective + "' in file: " + resultFile)
  exit(1)
 
 # Finding value for objective column
 try:
  objectiveValue = float(resultLines[-1].split()[objectiveCol])
 except ValueError:
  print("[Korali] Error: Could not read objective column position: '" + str(objectiveCol) + "' from the last line in file: " + resultFile)
  exit(1)
  
 # Trying to find t column
 try:
  tCol = columnNames.index('t')
 except ValueError:
  print("[Korali] Error: Could not load column name: 't' in file: " + resultFile)
  exit(1)
 
 # Finding value for t column
 try:
  t = float(resultLines[-1].split()[tCol])
 except ValueError:
  print("[Korali] Error: Could not read t column position: '" + str(tCol) + "' from the last line in file: " + resultFile)
  exit(1)
  
 # Trying to find dt column
 try:
  dtCol = columnNames.index('dt')
 except ValueError:
  print("[Korali] Error: Could not load column name: 'dt' in file: " + resultFile)
  exit(1)
 
 # Finding value for dt column
 try:
  dt = float(resultLines[-1].split()[dtCol])
 except ValueError:
  print("[Korali] Error: Could not read dt column position: '" + str(dtCol) + "' from the last line in file: " + resultFile)
  exit(1)
  
 # Checking whether the simulated time met the required tmax or was truncated
 if (t + dt < tmax):
  print("[Korali] Error: Simulation time (" + str(t + dt) + ") has not reached tmax (" + str(tmax) + ").")
  exit(1) 
 
 # Assigning objective function value
 x["F(x)"] = objectiveValue

 # Move back to the base directory
 os.chdir( CurrentDirectory )
