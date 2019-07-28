#!/usr/bin/env python3

import os
import sys

def buildSetup(koraliDir):
 print('[Korali] Building installation setup...')
 with open(koraliDir + '/.setup.py', 'r') as file: setupString = file.read()
 installFiles = ['libkorali.so', 'libkorali.a', '../libkorali.so']
 for dirpath, subdirs, files in os.walk(koraliDir):
  for x in files:
   if (x.endswith(".hpp") or x.endswith(".py")):
    installFiles.append(os.path.join(dirpath, x))
 setupString = setupString.replace('installFiles', str(installFiles))
 with open(koraliDir + '/setup.py', 'w') as file: file.write(setupString)
 
  