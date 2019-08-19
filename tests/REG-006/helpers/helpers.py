#!/usr/bin/env python3
import math

def evaluateModel( s ):
   x1 = s.getVariable(0)
   x2 = s.getVariable(1)
   r = -x1**2-x2**2-math.sin(x1)**2-math.sin(x2)**2
   s.setResult(r)

def inactive1( k ):
  k.setConstraintEvaluation(-1)

def inactive2( k ):
  k.setConstraintEvaluation(-2)

def activeMax1( k ):
    x = k.getVariables()
    c = -(x[0] - 1.0)
    k.setConstraintEvaluation(c)

def activeMax2( k ):
    x = k.getVariables()
    c = -(x[0] - 2.0)
    k.setConstraintEvaluation(c)

def activeMax3( k ):
    x = k.getVariables()
    c = -(x[1] - 1.0)
    k.setConstraintEvaluation(c)

def activeMax4( k ):
    x = k.getVariables()
    c = -(x[1] - 2.0)
    k.setConstraintEvaluation(c)

def inactiveMax1( k ):
    x = k.getVariables()
    c = -math.cos(x[0])
    k.setConstraintEvaluation(c)

def inactiveMax2( k ):
    x = k.getVariables()
    c = -math.sin(x[0])
    k.setConstraintEvaluation(c)

def inactiveMax3( k ):
    x = k.getVariables()
    c = -math.cos(x[1])
    k.setConstraintEvaluation(c)

def inactiveMax4( k ):
    x = k.getVariables()
    c = -math.sin(x[1])
    k.setConstraintEvaluation(c)

def stress1( k ):
    x = k.getVariables()
    c = -x[0] + 6.2
    k.setConstraintEvaluation(c)

def stress2( k ):
    x = k.getVariables()
    c = x[0] - x[1]
    k.setConstraintEvaluation(c)

def stress3( k ):
    x = k.getVariables()
    c = x[0] + 2.0 - 2.0*x[1] 
    k.setConstraintEvaluation(c)

def stress4( k ):
    x = k.getVariables()
    c = 2*x[0] - 3*x[1]
    k.setConstraintEvaluation(c)

def stress5( k ):
    x = k.getVariables()
    c = -(x[0] - 6.28)*(x[1] - 6.28)
    k.setConstraintEvaluation(c)

def stress6( k ):
    x = k.getVariables()
    c = -math.cos(x[0])*math.cos(x[1])
    k.setConstraintEvaluation(c)

def stress7( k ):
    x = k.getVariables()
    c = -math.sin(x[0])*math.sin(x[1])
    k.setConstraintEvaluation(c)

def stress8( k ):
    x = k.getVariables()
    c = x[0] - x[1]**2
    k.setConstraintEvaluation(c)

def assert_value( val, expected ):
    assert val  == expected, "Error: Value should be {0} but is {1}\n".format(expected, val)

def assert_smallereq( smaller, greater ):
    assert greater >= smaller, "Error: Value {0} should be smaller {1}\n".format(smaller, greater)

def assert_greatereq( greater, smaller ):
    assert smaller <= greater, "Error: Value {0} should be greater {1}\n".format(greater, smaller)
