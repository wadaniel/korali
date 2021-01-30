#! /usr/bin/env python3
import argparse
import os
import json
from pathlib import Path


from variables import getCXXVariableName, getVariableType

variableHeaderTemplateFile = '/Users/garampat/work/ETH-work/projects/korali/source/variable'
configFileList = []
modulesDir = '../../source/modules/'
for moduleDir, relDir, fileNames in os.walk( modulesDir ):
    for fileName in fileNames:
        if '.config' in fileName:
          _name = os.path.splitext(fileName)[0]
          configFileList.append(os.path.join(moduleDir,_name+'.config'))


variableDeclarationString = ''
variableDeclarationSet = set()

for _file in configFileList:
  p = Path(_file)
  p = p.resolve()

  moduleConfig = json.loads( p.read_text() )

  if 'Variables Configuration' in moduleConfig:
    for v in moduleConfig["Variables Configuration"]:
      varName = getCXXVariableName(v["Name"])
      if (not varName in variableDeclarationSet):
        variableDeclarationString += '/**\n'
        variableDeclarationString += '* @brief [Module: ' + moduleConfig["Module Data"]["Class Name"] + '] ' + v["Description"] + '\n'
        variableDeclarationString += '*/\n'
        variableDeclarationString += '  ' + getVariableType(v) + ' ' + varName + ';\n'
        variableDeclarationSet.add(varName)
