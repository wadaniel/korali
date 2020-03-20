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
hppIncludeLine = '#include "modules/' + moduleType + '/' + moduleType + '.hpp"'
cppIncludeLine = '#include "modules/' + moduleType + '/' + moduleName + '/' + moduleName + '.hpp"'
namespaceLine = 'namespace korali { namespace ' + moduleType + ' {'
ifdefTag = moduleName.upper()
parentClassName = 'korali::' + upcase_first_letter(moduleType)
moduleClassName = 'korali::' + upcase_first_letter(moduleName)

print(modulePath)
print(hppIncludeLine)
print(cppIncludeLine)
print(namespaceLine)
print(ifdefTag)
print(parentClassName)
print(moduleClassName)

mdevDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

print(mdevDir)

if __name__ == '__main__':
    main()
