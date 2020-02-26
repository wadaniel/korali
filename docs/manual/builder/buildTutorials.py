#!/usr/bin/env python3

import sys
import os
import json
import shutil

def processTutorialSubFolder(folder):
 os.makedirs('../tutorials/' + folder)
 tutorialsDir = '../../../tutorials/' + folder
 with open(tutorialsDir + '/README.rst', 'r') as file: tutorialsString = file.read()
 currentTutorial = 0
 
 # Detecting Tutorials
 tutorialNames  = os.listdir(tutorialsDir)
 for tutorialName in sorted(tutorialNames):
  tutorialPath =  tutorialsDir + '/' + tutorialName
  if (os.path.isdir(tutorialPath)):
   tutorialReadmeFile = tutorialsDir + '/' + tutorialName + '/README.rst'
   with open(tutorialReadmeFile, 'r') as file: tutorialReadmeString = file.read()
   
   with open(tutorialReadmeFile, 'r') as file:
    tutorialTitle = file.readline().replace('\n','')
    tutorialDescription = ''
    currentLineText = file.readline()
    while (not 'Example Scripts' in currentLineText):  
     currentLineText = file.readline()
     tutorialDescription += '     ' + currentLineText
    tutorialDescription = tutorialDescription.replace('Example Scripts', '')
      
   tutorialDstPath = folder + '/' + tutorialName + '.rst'
   
   tutorialsString += tutorialTitle + '\n'
   tutorialsString += '-' * len(tutorialTitle) + '\n\n'
   if ('examples' in tutorialsDir): tutorialsString += '   + **Source:** `https://github.com/cselab/korali-apps/tree/master/' + tutorialName + '/ <https://github.com/cselab/korali-apps/tree/master/' + tutorialName + '/>`_\n'
   else: tutorialsString += '   + **Source:** `https://github.com/cselab/korali/tree/master/tutorials/' + folder + '/' + tutorialName + '/ <https://github.com/cselab/korali/tree/master/tutorials/' + folder + '/' + tutorialName + '/>`_\n'
   tutorialsString += '   + **Description:** ' + tutorialDescription
   
   currentTutorial = currentTutorial + 1

 basicTutorialsFile = open('../tutorials/' + folder + '.rst', 'w')
 basicTutorialsFile.write(tutorialsString)
 basicTutorialsFile.close()
  
# Remaking tutorials folder
shutil.rmtree('../tutorials', ignore_errors=True, onerror=None)
os.makedirs('../tutorials')

processTutorialSubFolder('basic')
processTutorialSubFolder('advanced')
processTutorialSubFolder('examples')

