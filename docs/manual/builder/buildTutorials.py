#!/usr/bin/env python3

import sys
import os
import json
import glob
import shutil

def processTutorialSubFolder(folder):
 os.makedirs('../tutorials/' + folder)
 tutorialsDir = '../../../tutorials/' + folder
 with open(tutorialsDir + '/README.rst', 'r') as file: tutorialsString = file.read()

 tutorialsString += '\n\n'
 tutorialsString += ' .. toctree::\n'
 tutorialsString += '    :titlesonly:\n'
 tutorialsString += '    :maxdepth: 1\n\n'
 
 # Detecting Tutorials
 tutorialNames  = os.listdir(tutorialsDir)
 for tutorialName in sorted(tutorialNames):
  tutorialPath =  tutorialsDir + '/' + tutorialName
  if (os.path.isdir(tutorialPath)):
   tutorialReadmeFile = tutorialsDir + '/' + tutorialName + '/README.rst'
   with open(tutorialReadmeFile, 'r') as file: tutorialReadmeString = file.read()
   tutorialTitle = tutorialReadmeString.partition('\n')[0]
   tutorialDstPath = folder + '/' + tutorialName + '.rst'
   tutorialsString += '    ' + folder + '/' + tutorialName + '\n'
   tutorialOutputString = '.. _' + ''.join(i for i in tutorialName.replace('.','') if not i.isdigit()) + ':\n\n'
   tutorialOutputString += '**Tutorial Source Code:** `https://github.com/cselab/korali/tree/master/tutorials/' + folder + '/' + tutorialName + '/ <https://github.com/cselab/korali/tree/master/tutorials/' + folder + '/' + tutorialName + '/>`_\n\n'
   tutorialOutputString += tutorialReadmeString
   with open('../tutorials/' + tutorialDstPath, 'w') as file: tutorialReadmeString = file.write(tutorialOutputString)
   
   for file in glob.glob(r'' + tutorialsDir + '/' + tutorialName + '/*.png'):
    shutil.copy(file, '../tutorials/' + folder)

 basicTutorialsFile = open('../tutorials/' + folder + '.rst', 'w')
 basicTutorialsFile.write(tutorialsString)
 basicTutorialsFile.close()
  
# Remaking tutorials folder
shutil.rmtree('../tutorials', ignore_errors=True, onerror=None)
os.makedirs('../tutorials')

processTutorialSubFolder('basic')
processTutorialSubFolder('advanced')
processTutorialSubFolder('examples')