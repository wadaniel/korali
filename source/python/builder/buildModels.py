import sys
import os
import json
from buildAux import *

def buildModels(koraliDir):
 # Processing Models
 curdir = koraliDir + '/models' 
 
 # Detecting Models
 modelPaths  = [x[0] for x in os.walk(curdir)][1:]
 for modelPath in modelPaths:
  modelName = modelPath.replace(curdir + '/', '')
  
  ####### Producing model.hpp
  
  # Loading template header .hpp file
  modelTemplateHeaderFile = modelPath + '/' + modelName + '._hpp'
  with open(modelTemplateHeaderFile, 'r') as file: modelTemplateHeaderString = file.read()
  
  newHeaderString = modelTemplateHeaderString
  
  # Saving new header .hpp file
  modelNewHeaderFile = modelPath + '/' + modelName + '.hpp'
  print('[Korali] Creating: ' + modelNewHeaderFile + '...')
  with open(modelNewHeaderFile, 'w') as file: file.write(newHeaderString)
