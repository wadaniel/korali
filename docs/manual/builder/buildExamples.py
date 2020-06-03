#!/usr/bin/env python3

import sys
import os
import json
import shutil
import copy

exampleSrcDir = '../../../examples'

################################################
# Process Example Function

def processExample(exampleRelPath, exampleName):
  examplePath = os.path.join(exampleSrcDir, exampleRelPath)
  exampleReadmeFile = examplePath + '/README.rst'
  exampleOutputDir = os.path.abspath(os.path.join('../examples/' + exampleRelPath, os.pardir)) 

  print('Processing file: ' + exampleReadmeFile)

  exampleReadmeString = '.. _example_' + exampleRelPath.lower().replace('./', '').replace('/', '-').replace(' ', '') + ':\n\n'

  # Adding source code
  exampleReadmeString += '.. hint::\n\n'
  exampleReadmeString += '   Example source code: `https://github.com/cselab/korali/tree/master/tutorials/' + exampleRelPath.replace('./','') + '/ <https://github.com/cselab/korali/tree/master/tutorials/' + exampleRelPath.replace('./','') + '/>`_\n\n'

  # Reading original rst
  with open(exampleReadmeFile, 'r') as file:
   exampleReadmeString += file.read() + '\n\n'

  # Creating subfolder list
  subFolderList = []
  list_dir = os.listdir(examplePath)
  for f in list_dir:
    fullPath = os.path.join(examplePath, f)
    if not os.path.isfile(fullPath):
      if (not '.o/' in fullPath and not '.d/' in fullPath and not '/_' in fullPath):
        subFolderList.append(f)

   # Creating example's folder, if not exists
  if not os.path.exists(exampleOutputDir):
    os.mkdir(exampleOutputDir)
         
  # Determining if its a parent or leaf example
  isParentExample = True
  if (subFolderList == []):
    isParentExample = False

  # If its parent, construct children examples
  if (isParentExample == True):
    exampleReadmeString += '**Sub-Categories**\n\n'
    exampleReadmeString += '.. toctree::\n'
    exampleReadmeString += '   :titlesonly:\n\n'

    for f in subFolderList:
      subExampleFullPath = os.path.join(examplePath, f)
      if (not '/_' in subExampleFullPath ):
       exampleReadmeString += '   ' + f + '/' + f + '\n'
       subPath = os.path.join(exampleRelPath, f)
       processExample(subPath, f)

  # If its leaf, build actual readme with links
  if (isParentExample == False):
    exampleReadmeString += ''

  # Saving Example's readme file
  exampleReadmeString += '\n\n'
  with open(exampleOutputDir + '/' + exampleName + '.rst', 'w') as file:
    file.write(exampleReadmeString)


############################################
# Main Procedure

shutil.rmtree('../examples', ignore_errors=True, onerror=None)
os.makedirs('../examples')

list_dir = os.listdir(exampleSrcDir)
for f in list_dir:
  fullPath = os.path.join(exampleSrcDir, f)
  if not os.path.isfile(fullPath):
    if (not '.o/' in fullPath and not '.d/' in fullPath and not '/_' in fullPath):
      processExample(f, f)
