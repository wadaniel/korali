#!/usr/bin/env python3
import sys
import subprocess

def runModel(x):
  argString = ['model/model.py']
  v = x.getVariables()
  for i in v: argString.append(str(i))
  retValue = subprocess.check_output(argString)
  x.setResult(float(retValue.decode()))

