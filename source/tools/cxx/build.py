#!/usr/bin/env python3

import os
import sys

def buildCxx(koraliDir):
 print('[Korali] Building C++ compilation commands...')
 makefileConfFile = open(koraliDir + '/Makefile.conf', 'r')
 CXXFLAGS=''
 for line in makefileConfFile:
   CXXFLAGS += line.replace('=', '=\'', 1).replace('\n', '') + '\'\n'
 
 with open(koraliDir + '/python/cxx/cflags._py', 'r') as file: cFlagsString = file.read()
 with open(koraliDir + '/python/cxx/compiler._py', 'r') as file: compilerString = file.read()
 with open(koraliDir + '/python/cxx/libs._py', 'r') as file: libString = file.read()
 with open(koraliDir + '/python/cxx/cflags.py', 'w+') as file: file.write(CXXFLAGS + '\n' + cFlagsString)
 with open(koraliDir + '/python/cxx/compiler.py', 'w+') as file: file.write(CXXFLAGS + '\n' + compilerString)
 with open(koraliDir + '/python/cxx/libs.py', 'w+') as file: libString = file.write(CXXFLAGS + '\n' + libString)
 
