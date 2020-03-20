#! /usr/bin/env python3
import os
import sys
import re
import argparse

def upcase_first_letter(s):
    return s[0].upper() + s[1:] 
    
if __name__ == '__main__':
 parser = argparse.ArgumentParser(prog='korali.mdev', description='Creates a new module for Korali.')
 parser.add_argument('--type', help='Module type to create', required = True, choices=[ 'problem', 'solver' ])
 parser.add_argument('--name', help='Name of the new module.', required = True)
 args = parser.parse_args()

 checkModuleFile = './source/modules/module._hpp'
 if (not os.path.isfile(checkModuleFile)):
  print('[Error] This module can only be run at Korali\'s base folder (did not find file: ' + checkModuleFile + ' ...).')
  exit(-1)
 
 moduleType = args.type
 moduleName = args.name

 if (not re.match("[A-Za-z0-9_-]*$", moduleName)):
  print('Error: the module\'s name can only contain letters, numbers, underscores, and dashes')
  exit(-1)

 modulePath = './source/modules/' + moduleType + '/' + moduleName
 moduleClassName = upcase_first_letter(moduleName)
 moduleTag = moduleName.upper()
 baseExampleDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/' + moduleType

 with open(baseExampleDir + '/base.cpp', 'r') as file: cppBaseString = file.read()
 with open(baseExampleDir + '/base.hpp', 'r') as file: hppBaseString = file.read()
 with open(baseExampleDir + '/base.config', 'r') as file: configBaseString = file.read()
 with open(baseExampleDir + '/README.rst', 'r') as file: readmeBaseString = file.read()

 if (os.path.exists(modulePath)):
  print('Error: The specified module already exists: ' + modulePath)
  exit(-1)
  
 os.mkdir(modulePath)

 cppString = cppBaseString.replace('[CLASSNAME]', moduleClassName).replace('[MODULENAME]', moduleName)
 hppString = hppBaseString.replace('[CLASSNAME]', moduleClassName).replace('[MODULETAG]', moduleTag)
 configString = configBaseString
 readmeString = readmeBaseString.replace('[MODULENAME]', moduleName)
 
 with open(modulePath + '/' + moduleName + '._cpp', 'w') as file: file.write(cppString)
 with open(modulePath + '/' + moduleName + '._hpp', 'w') as file: file.write(hppString)
 with open(modulePath + '/' + moduleName + '.config', 'w') as file: file.write(configString)
 with open(modulePath + '/README.rst', 'w') as file: file.write(readmeString)
 
 print('[Success] Created new module: ' + modulePath)