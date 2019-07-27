import os
extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 

import sys
sys.path.append(extdir)
from libkorali import *
 
def initialize():
 return Engine()
 
def getResults(fileName = ''):
 return Engine.getResults(fileName)
