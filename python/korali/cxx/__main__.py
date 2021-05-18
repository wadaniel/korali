#! /usr/bin/env python3
import os
import sys
import sysconfig
import subprocess
import korali
import pybind11
import libkorali as lk

def main():

  if (len(sys.argv) != 2):
    print('[Korali] Syntax error on call to korali.cxx module: Exactly one argument is required (--cflags, --libs).')
    exit(-1)
  correctSyntax = False
  
  # Looking for Korali library
  koraliLib = os.path.realpath(lk.__file__)

  if (sys.argv[1] == '--cflags'):
    correctSyntax = True
    pythonCFlagsCommand = subprocess.Popen("python3-config --includes", shell=True, stdout=subprocess.PIPE)
    pythonCFlags = pythonCFlagsCommand.stdout.read().decode().rstrip("\n")
    pybind11Includes=pybind11.get_include()
    koraliIncludes=' -I' + os.path.dirname(korali.__file__) + '/include' + ' -I' +  os.path.dirname(korali.__file__) + '/../../../../include/' + ' -I' + os.path.dirname(korali.__file__) + '/../../source/'
    flags = '-std=c++17' + koraliIncludes + ' -I' + sysconfig.get_path("include") + ' -I' + sysconfig.get_path("platinclude") + ' -I' + pybind11Includes + ' ' + pythonCFlags

  if (sys.argv[1] == '--libs'):
    correctSyntax = True
    
    pythonPrefixCommand = subprocess.Popen("python3-config --prefix", shell=True, stdout=subprocess.PIPE)
    pythonPrefix = pythonPrefixCommand.stdout.read().decode().rstrip("\n")

    try:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags --embed", shell=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags", shell=True, stdout=subprocess.PIPE)
    pythonLibs = pythonLibsCommand.stdout.read().decode().rstrip("\n")
    try:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags --embed", shell=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags", shell=True, stdout=subprocess.PIPE)
    pythonLibs = pythonLibsCommand.stdout.read().decode().rstrip("\n")
    
    flags = koraliLib + ' -Wl,-rpath,' + os.path.dirname(korali.__file__) + ' ' + pythonLibs + ' -Wl,-rpath,' + pythonPrefix + '/lib'

  if (correctSyntax == False):
   print('[Korali] Syntax error on call to korali.cxx module: Argument \'{0}\' not recognized (--cflags, --libs).'.format(sys.argv[1]))
   exit(-1)

  print(flags + ' ')

if __name__ == '__main__':
  main()
