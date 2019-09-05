import os
extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 
from korali.auxiliars.fileIO import * 

import sys
sys.path.append(extdir)
 
def initialize():
 from libkorali import Engine
 return Engine()
 
def getResults(fileName = ''):
 return Engine.getResults(fileName)
 

def getLatestResult(src):
 resultFiles = getResultFiles(src)
 if (not resultFiles): return ''
 return src + '/' + resultFiles[-1]
 
