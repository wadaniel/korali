#!/usr/bin/env python3
import math

def evaluateModel( s ):
   x1 = s.getVariable(0)
   x2 = s.getVariable(1)
   r = -x1**2*x2**2-math.sin(x1)**2-math.sin(x2)**2
   s.setResult(r)

def inactive( k ):
  k.setConstraintEvaluation(-10);

def assert_value( val, expected ):
    assert val  == expected, "Error: Value should be {0} but is {1}\n".format(expected, val)

def assert_smallereq( smaller, greater ):
    assert greater >= smaller, "Error: Value {0} should be smaller {1}\n".format(smaller, greater)

def assert_greatereq( greater, smaller ):
    assert smaller <= greater, "Error: Value {0} should be greater {1}\n".format(greater, smaller)
