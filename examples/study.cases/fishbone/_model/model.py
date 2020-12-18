#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil

def model(x):
 
 jobId = 0
 if 'SLURM_JOBID' in os.environ: jobId = os.environ['SLURM_JOBID']

 SourceFolderName = "_config"
 DestinationFolderName = '_results/job' + str(jobId) + '/' + 'sample' + str(x["Sample Id"])
 
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
  f.write('angle = %.10f\n' % x["Parameters"][0] )
  f.write('bone_factor = %.10f\n' % x["Parameters"][1] )
  
 # Run Aphros for this sample
 subprocess.call("make")

 # Read the Loglikelihood value
 Y = readColumnFromFile(ResultsFile, 0)

 x["F(x)"] = 0

 # Move back to the base directory
 os.chdir( CurrentDirectory )

 # Delete the temporary directory -- If necessary
 #if os.path.exists( DestinationFolderName ):
 # shutil.rmtree( DestinationFolderName)

