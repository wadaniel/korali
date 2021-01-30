#!/usr/bin/env python3
import argparse
import os

from builders import *


# parser = argparse.ArgumentParser()
# parser.add_argument('--dir', '-d', default='./', help='Korali root directory')
# args = parser.parse_args()

configFileList = []
headerFileList = []
sourceFileList = []
modulesDir = '../../source/modules'
for moduleDir, relDir, fileNames in os.walk( modulesDir ):
    for fileName in fileNames:
        if '.config' in fileName:
          _name = os.path.splitext(fileName)[0]
          configFileList.append(os.path.abspath(os.path.join(moduleDir,_name+'.config')))
          headerFileList.append(os.path.abspath(os.path.join(moduleDir,_name+'._hpp')))
          sourceFileList.append(os.path.abspath(os.path.join(moduleDir,_name+'._cpp')))

print()

for configFile, headerFile, sourceFile in zip(configFileList, headerFileList, sourceFileList):
  print(configFile)
  buildCodeFromTemplate( configFile, headerFile )
  buildCodeFromTemplate( configFile, sourceFile )
