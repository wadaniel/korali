import os
extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 
from korali.auxiliars.fileIO import * 

import sys
sys.path.append(extdir)
from libkorali import *
 
def initialize():
 return Engine()
 
def getResults(fileName = ''):
 return Engine.getResults(fileName)
 

def getLatestResult(src):
 resultFiles = getResultFiles(src)
 return src + '/' + resultFiles[-1]
 