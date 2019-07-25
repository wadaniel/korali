#! /usr/bin/env python3
import os
import sys
from korali.cxx.cflags import getcflags
from korali.cxx.libs import getlibs
from korali.cxx.compiler import getcompiler

def main():
 extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
 
 correctSyntax=False
 
 if (sys.argv[1] == '--cflags'):
   correctSyntax=True
   getcflags(extdir)
   
 if (sys.argv[1] == '--libs'):
   correctSyntax=True
   getlibs(extdir)
   
 if (sys.argv[1] == '--compiler'):
   correctSyntax=True
   getcompiler(extdir)
  
 if (correctSyntax==False):
   print('[Korali] Syntax error on call to korali.cxx module.')
   exit -1 
  
if __name__ == '__main__':
    main()
