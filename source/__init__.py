import os
extdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

import sys
sys.path.append(extdir)


def Engine():
  from libkorali import Engine
  return Engine()


def Experiment():
  from libkorali import Experiment
  return Experiment()


def getMPICommPtr():
  from libkorali import getMPICommPointer
  return getMPICommPointer()

def getMPIComm():
  from libkorali import getMPI4PyComm
  return getMPI4PyComm()