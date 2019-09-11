#!/usr/bin/env python3
import sys
import subprocess

def runModel(x):
  argString = ['model/model.py']
  v = x.getSampleData()
  for i in v: argString.append(str(i))
  retValue = subprocess.check_output(argString)
  result = float(retValue.decode())
  x.setResult(result)

