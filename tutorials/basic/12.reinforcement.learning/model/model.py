#!/usr/bin/env python

def runEnvironment(p):
  print('Environment Initialized')
  
  for i in range(10):
   p['State'] = [ 0.0 ]
   p.update()
   print('Performing Action: ' + str(i))