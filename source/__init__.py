import os
extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 

import sys
sys.path.append(extdir)
 
def initialize():
 from libkorali import Korali
 return Korali()

def newExperiment():
 from libkorali import Engine
 return Engine()
 
