#! /usr/bin/env python3
import os
import sys

pathCheckFile = './source/modules/module._hpp'
if (not os.path.isfile(pathCheckFile)):
 print('[Error] Could not find Korali source files, are you currently at Korali\'s base directory?')
 exit(-1)

print('Found File')
