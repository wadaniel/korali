#!/usr/bin/env python3

import os
import sys

def buildCXX():
 print('[Korali] Building C++ compilation commands...')
 curdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 
 makefileConfFile = open(curdir + '/../../Makefile.conf', 'r')
 CXXFLAGS=''
 for line in makefileConfFile:
   CXXFLAGS += line.replace('=', '=\'', 1).replace('\n', '') + '\'\n'
 
 with open(curdir + '/cflags.py.base', 'r') as file: cFlagsString = file.read()
 with open(curdir + '/compiler.py.base', 'r') as file: compilerString = file.read()
 with open(curdir + '/libs.py.base', 'r') as file: libString = file.read()
 with open(curdir + '/cflags.py', 'w+') as file: file.write(CXXFLAGS + '\n' + cFlagsString)
 with open(curdir + '/compiler.py', 'w+') as file: file.write(CXXFLAGS + '\n' + compilerString)
 with open(curdir + '/libs.py', 'w+') as file: libString = file.write(CXXFLAGS + '\n' + libString)
 
