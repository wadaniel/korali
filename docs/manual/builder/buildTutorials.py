#!/usr/bin/env python3

import sys
import os
import json
import shutil

def processTutorialSubFolder(folder):
 os.makedirs('../tutorials/' + folder)
 tutorialsDir = '../../../tutorials/' + folder
 with open(tutorialsDir + '/README.rst', 'r') as file: tutorialsString = file.read()

 # Detecting Tutorials
 tutorialNames  = os.listdir(tutorialsDir)
 for tutorialName in sorted(tutorialNames):
  tutorialPath =  tutorialsDir + '/' + tutorialName
  if (os.path.isdir(tutorialPath)):
   tutorialReadmeFile = tutorialsDir + '/' + tutorialName + '/README.rst'
   with open(tutorialReadmeFile, 'r') as file: tutorialReadmeString = file.read()
   tutorialTitle = tutorialReadmeString.partition('\n')[0]
   tutorialDstPath = folder + '/' + tutorialName + '.rst'
   tutorialsString += '   ' + folder + '/' + tutorialName + '\n'
   shutil.copyfile(tutorialReadmeFile, '../tutorials/' + tutorialDstPath)

 basicTutorialsFile = open('../tutorials/' + folder + '.rst', 'w')
 basicTutorialsFile.write(tutorialsString)
 basicTutorialsFile.close()
  
# Remaking tutorials folder
shutil.rmtree('../tutorials', ignore_errors=True, onerror=None)
os.makedirs('../tutorials')

processTutorialSubFolder('basic')
processTutorialSubFolder('advanced')
processTutorialSubFolder('examples')

