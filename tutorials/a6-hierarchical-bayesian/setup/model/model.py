#!/usr/bin/env python3
import sys
import math

def logistic( X, s ):
  th1 = s.getVariable(0)
  th2 = s.getVariable(1)
  th3 = s.getVariable(2)

  for x in X:
    f = math.exp(th3*x)
    y = ( th1*th2*f )/( th1 + th2*(f-1) )
    s.addReferenceEvaluation(y)


def getReferenceData( path, i ):
  fileName = path + "/data_set_" + str(i).zfill(3) + ".dat"
  y = readColumnFromFile(fileName,1)
  return y


def getReferencePoints( path, i ):
  fileName = path + "/data_set_" + str(i).zfill(3) + ".dat"
  y = readColumnFromFile(fileName,0)
  return y


def readColumnFromFile( FileName, Column ):
  try:
    f = open( FileName, "r" )
    lines = f.readlines()
    y = []
    for l in lines:
      y.append(float(l.split()[Column]))
    f.close()
  except IOError as e:
    print("I/O error(" + str(e.errno) + "): " + e.strerror )
    print("The file " + FileName + " is missing")
    sys.exit(1)
  except ValueError:
    print( "Could not convert data to a float. Check the results in " + FileName )
    sys.exit(1)
  except:
    print( "Unexpected error: " + sys.exc_info()[0] )
    raise

  return y

