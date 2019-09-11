#!/usr/bin/env python3
import math

def evaluateModel( s ):
   x1 = s[0]
   x2 = s[1]
   r = -x1**2-x2**2-math.sin(x1)**2-math.sin(x2)**2
   s.setResult(r)

def inactive1( k ):
  k.setResult(-1)

def inactive2( k ):
  k.setResult(-2)

def activeMax1( k ):
    c = -(k[0] - 1.0)
    k.setResult(c)

def activeMax2( k ):
    c = -(k[0] - 2.0)
    k.setResult(c)

def activeMax3( k ):
    c = -(k[1] - 1.0)
    k.setResult(c)

def activeMax4( k ):
    c = -(k[1] - 2.0)
    k.setResult(c)

def inactiveMax1( k ):
    c = -math.cos(k[0])
    k.setResult(c)

def inactiveMax2( k ):
    c = -math.sin(k[0])
    k.setResult(c)

def inactiveMax3( k ):
    c = -math.cos(k[1])
    k.setResult(c)

def inactiveMax4( k ):
    c = -math.sin(k[1])
    k.setResult(c)

def stress1( k ):
    c = -k[0] + 6.2
    k.setResult(c)

def stress2( k ):
    c = k[0] - k[1]
    k.setResult(c)

def stress3( k ):
    c = k[0] + 2.0 - 2.0*k[1] 
    k.setResult(c)

def stress4( k ):
    c = 2*k[0] - 3*k[1]
    k.setResult(c)

def stress5( k ):
    c = -(k[0] - 6.28)*(k[1] - 6.28)
    k.setResult(c)

def stress6( k ):
    c = -math.cos(k[0])*math.cos(k[1])
    k.setResult(c)

def stress7( k ):
    c = -math.sin(k[0])*math.sin(k[1])
    k.setResult(c)

def stress8( k ):
    c = k[0] - k[1]**2
    k.setResult(c)

def assert_value( val, expected ):
    assert val  == expected, "Error: Value should be {0} but is {1}\n".format(expected, val)

def assert_smallereq( smaller, greater ):
    assert greater >= smaller, "Error: Value {0} should be smaller {1}\n".format(smaller, greater)

def assert_greatereq( greater, smaller ):
    assert smaller <= greater, "Error: Value {0} should be greater {1}\n".format(greater, smaller)
