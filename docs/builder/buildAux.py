#!/usr/bin/env python3

import sys
import os
import json

#####################################################################

def getVariableName(v):
 cVarName = v["Name"].replace(" ", "")
 cVarName = '_' + cVarName[0].lower() + cVarName[1:]
 return cVarName
 
def getVariableDefault(v):
 return v.get('Default', '*none*')

def getJsonPath(path):
 nameString = ''
 for item in path:
  nameString += '["' + item + '"]'
 return nameString
 
def getVariableInfo(v, moduleName):
 varString = ''
 varString += '??? abstract "' + getJsonPath(v["Name"]) + '"\n\n'
 varString += '\t' + v["Description"] + '\n'
 varString += '\n'
 
 if ('Korali::Distribution' in v["Type"]):
  with open('distributionList.txt', 'r') as file: distributionListString = file.read()
  varString += '\t**Available Distributions**: \n'
  varString += distributionListString
  return varString
  
 varString += '\t+ Default Value: ' + getVariableDefault(v) + '\n'
 varString += '\t+ Datatype: ' + v["Type"] + '\n'
 varString += '\t+ Syntax: \n\n' 
 varString += '\t```python\n\t\tkorali["Variables"][i]["' + moduleName + '"]["' + getJsonPath(v["Name"]) + '"] = *value*\n\t```\n\n'
 return varString

##################################################################### 