#!/usr/bin/env python3
import sys
import subprocess

def runModel(x):
  argString = ['model/model.py']
  for i in range(x.getVariableCount()):
    argString.append(str(x.getVariable(i)))
  retValue = subprocess.check_output(argString)
  x.addResult(float(retValue.decode()))

