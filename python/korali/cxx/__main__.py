#! /usr/bin/env python3
import os
import sys
import ast
import sysconfig
import glob, os

def main():
  fileDir = os.path.dirname(os.path.realpath(__file__))
  koraliDir = os.path.abspath(fileDir + '/../')

  configFile = open(fileDir + '/flags.in', "r")
  configFileContents = configFile.read()
  configDict = ast.literal_eval(configFileContents)
  configFile.close()

  if (len(sys.argv) != 2):
    print('[Korali] Syntax error on call to korali.cxx module: Exactly one argument is required (--cflags, --libs, or --compiler).')
    exit(-1)
        
  if (sys.argv[1] == '--cflags'):
    correctSyntax = True
    flags = '-I' + koraliDir + '/include' + ' -I' + sysconfig.get_path("include") + ' -I' + sysconfig.get_path("platinclude") + ' -I' + configDict['PYBIND11_INCLUDES']

  if (sys.argv[1] == '--libs'):
    correctSyntax = True
    
    # Looking for Korali library
    koraliLib = glob.glob(koraliDir + "/*.so")[0]
    flags = '-L' + koraliDir + ' -L' + koraliDir + '/../../../../lib' + ' -L' + koraliDir + '/../../../../lib64' + ' ' + koraliLib + ' -lpython'

  if (sys.argv[1] == '--compiler'):
    correctSyntax = True
    flags = configDict['CXX']

  if (correctSyntax == False):
   print('[Korali] Syntax error on call to korali.cxx module: Argument \'{0}\' not recognized (--cflags, --libs, or --compiler).'.format(sys.argv[1]))
   exit(-1)

  print(flags + ' ')

if __name__ == '__main__':
  main()
