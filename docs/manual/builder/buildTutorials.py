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
   tutorialTitle = tutorialReadmeString.partition('\n')[0]
   tutorialDstPath = folder + '/' + tutorialName + '.rst'
   
   tutorialsString += tutorialTitle + '\n'
   tutorialsString += '-' * len(tutorialTitle) + '\n\n'
   tutorialsString += '   + **Description:** Lorem ipsum dolor sit amet, consectetur adipiscing elit. In semper consequat purus eget sagittis. Aliquam erat volutpat. Morbi pulvinar nibh ante, ut laoreet lacus venenatis elementum. Aenean aliquet vitae massa vitae convallis. Nulla et velit tempor, facilisis massa nec, dapibus lorem. Aenean mauris justo, dignissim nec eros sit amet, ornare consectetur velit. Nullam suscipit nisi justo, vitae euismod orci dictum non.\n'
   tutorialsString += '   + **Source:** `https://github.com/cselab/korali/tree/master/tutorials/' + folder + '/' + tutorialName + '/ <https://github.com/cselab/korali/tree/master/tutorials/' + folder + '/' + tutorialName + '/>`_\n'
   tutorialsString += '\n\n'
   
   #tutorialsString += '\n\n'
   #tutorialTitleLabel   =  '|tutorial' + str(currentTutorial).zfill(3) + '|'
   #tutorialsString += ' .. ' + tutorialTitleLabel + ' replace::  `' + tutorialTitle + ' <https://github.com/cselab/korali/tree/master/tutorials/' + folder + '/' + tutorialName + '/>`__\n'
   #tutorialsString += ' .. table:: \n'
   #tutorialsString += '     :widths: 8 92\n\n'
   #tutorialsString += '     +----------------------+--------------------------------------------------------+\n'
   #tutorialsString += '     | **Title:**           | ' + tutorialTitleLabel + '                                          |\n'
   #tutorialsString += '     +----------------------+--------------------------------------------------------+\n'
   #tutorialsString += '     | **Description:**     |                                                        |\n'
   #tutorialsString += '     +----------------------+--------------------------------------------------------+\n'
   
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

