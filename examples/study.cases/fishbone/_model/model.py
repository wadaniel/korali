#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil

def model(x, resultFolder):
 
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
  f.write('angle = %.10f\n' % x["Parameters"][0] )
  f.write('bone_factor = %.10f\n' % x["Parameters"][1] )
  
 # Run Aphros for this sample
 sampleOutFile = open('sample.out', 'w')
 sampleErrFile = open('sample.err', 'w')
 subprocess.call(['bash', 'run.sh'], stdout=sampleOutFile, stderr=sampleErrFile)

 x["F(x)"] = 0

 # Move back to the base directory
 os.chdir( CurrentDirectory )

 # Delete the temporary directory -- If necessary
 #if os.path.exists( DestinationFolderName ):
 # shutil.rmtree( DestinationFolderName)

