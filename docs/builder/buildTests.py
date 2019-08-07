import sys
import os
import json
from buildAux import *

def buildTests(koraliDir):
 # Processing Tests
 testsDir = koraliDir + '/tests'
 
 # Loading installation page
 installMdFileName = koraliDir + '/docs/docs/install._md'
 with open(installMdFileName, 'r') as file: installString = file.read()
 
 # Creating test entry table
 testTableString = ''
 
 # Detecting Tests
 testPaths  = os.listdir(testsDir)
 for testPath in sorted(testPaths):
  testPath =  testsDir + '/' + testPath
  if (os.path.isdir(testPath)):
   testName = testPath.replace(testsDir + '/', '')
  
   ####### Producing test page
 
   testDocFile = testPath + '/README.md'
   with open(testDocFile, 'r') as file: testDocString = file.read()
   
   testDescription = testDocString.splitlines()[2]
   
   if ('REG' in testName): testType = 'Regression Test'
   if ('UNIT' in testName): testType = 'Unit Test'
   if ('STAT' in testName): testType = 'Statistical Test'
   
   testTableString += '| ' + testType + ' | [' + testName + '](https://github.com/cselab/korali/tree/master/tests/' + testName + ') | ' + testDescription + ' | \n' 
    
 ###### Saving tests web page
 
 installString = installString.replace('<!--- Test list goes here --->', testTableString)
 
 #webFileName = koraliDir + '/docs/docs/install.md'
 #print('[Korali] Creating ' + webFileName + '...')  
 #with open(webFileName, 'w+') as file: file.write(installString)
  