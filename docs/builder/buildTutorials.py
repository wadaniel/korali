import sys
import os
import json
from buildAux import *

def buildTutorials(koraliDir):
 # Processing Tutorials
 tutorialsDir = koraliDir + '/tutorials'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/tutorials/'
 if (not os.path.isdir(outputDir)): os.makedirs(outputDir)
 
 # Initializing Mkdocs entries string
 mkdocsEntriesString = ''
  
 # Detecting Tutorials
 tutorialPaths  = os.listdir(tutorialsDir)
 for tutorialPath in tutorialPaths:
  tutorialPath =  tutorialsDir + '/' + tutorialPath
  if (os.path.isdir(tutorialPath)):
   tutorialName = tutorialPath.replace(tutorialsDir + '/', '')
   
   # Adding yaml entry
   mkdocsEntriesString += '          - ' + tutorialName + ': usage/tutorials/' + tutorialName + '.md\n'
   
   ####### Producing tutorial page
   
   tutorialDocFile = tutorialPath + '/README.md'
   with open(tutorialDocFile, 'r') as file: tutorialDocString = file.read()
   
   sourceCodeString = '!!! info\n'
   sourceCodeString += '\tThe Python source for this tutorial can be found at:\n\n'
   sourceCodeString += '\t+ [' + tutorialName + '/python/](https://github.com/cselab/korali/tree/master/tutorials/' + tutorialName + '/python/)\n\n'
   sourceCodeString += '\tThe C++ source for this tutorial can be found at:\n\n'
   sourceCodeString += '\t+ [' + tutorialName + '/cxx/](https://github.com/cselab/korali/tree/master/tutorials/' + tutorialName + '/cxx/)\n\n'
   sourceCodeString += '\n'
   
   tutorialDocString = tutorialDocString.replace('## Source Code', '## Source Code\n\n' + sourceCodeString)
   
   mdFileName = outputDir + tutorialName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(tutorialDocString)

 return mkdocsEntriesString