import sys
import os
import json
from buildAux import *

def buildModels(koraliDir):
 # Processing Models
 modelsDir = koraliDir + '/source/models'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/manual/models/'
 if (not os.path.isdir(outputDir)): os.makedirs(outputDir)
 
 # Creating model list string
 modelListString = ''
 
 # Detecting Models
 modelPaths  = os.listdir(modelsDir)
 for modelPath in modelPaths:
  modelPath =  modelsDir + '/' + modelPath
  if (os.path.isdir(modelPath)):
   modelName = modelPath.replace(modelsDir + '/', '')
  
   ####### Producing model page
 
   modelDocFile = modelPath + '/README.md'
   with open(modelDocFile, 'r') as file: modelDocString = file.read()
   
   mdFileName = koraliDir + '/docs/docs/manual/models/' + modelName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(modelDocString)

   ####### Adding model list entry
   
   modelListString += '+ [' + modelName.capitalize() + ' Model](/manual/models/' + modelName + '.md)\n'
   
 ###### Returning Model list
 
 return modelListString