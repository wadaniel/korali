#! /usr/bin/env python3
import os
import sys
import re

def upcase_first_letter(s):
    return s[0].upper() + s[1:] 
    
def main():

 checkModuleFile = './source/modules/module._hpp'
 if (not os.path.isfile(checkModuleFile)):
  print('[Error] This module can only be run at Korali\'s base folder (did not find file: ' + checkModuleFile + ' ...).')
  exit(-1)
 
print('Korali Module Creation Tool')
print('===========================')
print('')

selectedType = False
while (selectedType == False):
 print('Choose Module type to create:')
 print('1) Problem')
 print('2) Solver')

 c = input("Choice: ")
 if (c == '1' or c == '2'): selectedType = True

if (c == '1'): moduleType = 'problem'
if (c == '2'): moduleType = 'solver'

selectedName = False
while (selectedName == False):
 moduleName = input("Enter new module's name: ")
 if (not re.match("[A-Za-z0-9_-]*$", moduleName)):
   print('Error: the module\'s name can only contain letters, numbers, underscores, and dashes')
 else: selectedName = True

modulePath = './source/modules/' + moduleType + '/' + moduleName
moduleClassName = upcase_first_letter(moduleName)
baseExampleDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/' + moduleType

with open(baseExampleDir + '/base.cpp', 'r') as file: cppBaseString = file.read()
with open(baseExampleDir + '/base.hpp', 'r') as file: hppBaseString = file.read()
with open(baseExampleDir + '/base.config', 'r') as file: configBaseString = file.read()
with open(baseExampleDir + '/README.rst', 'r') as file: readmeBaseString = file.read()

print(modulePath)
print(moduleClassName)
print(baseExampleDir)

#os.mkdir(modulePath)

if __name__ == '__main__':
    main()
