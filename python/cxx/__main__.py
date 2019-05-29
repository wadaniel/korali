#! /usr/bin/env python3
import os
import sys
from korali.cxx.cflags import getcflags

def main():
 extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
 
 if (sys.argv[1] == '--cflags'):
   getcflags(extdir)
  
if __name__ == '__main__':
    main()
