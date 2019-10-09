#!/usr/bin/env python3
import math

def evaluateModel( s ):
   x1 = s["Parameters"][0]
   x2 = s["Parameters"][1]
   r = -x1**2-x2**2-math.sin(x1)**2-math.sin(x2)**2
   s["Evaluation"] = r

def inactive1( k ):
  k["Evaluation"] = -1

def inactive2( k ):
  k["Evaluation"] = -2

def activeMax1( k ):
    c = -(k["Parameters"][0] - 1.0)
    k["Evaluation"] = c

def activeMax2( k ):
    c = -(k["Parameters"][0] - 2.0)
    k["Evaluation"] = c

def activeMax3( k ):
    c = -(k["Parameters"][1] - 1.0)
    k["Evaluation"] = c

def activeMax4( k ):
    c = -(k["Parameters"][1] - 2.0)
    k["Evaluation"] = c

def inactiveMax1( k ):
    c = -math.cos(k["Parameters"][0])
    k["Evaluation"] = c

def inactiveMax2( k ):
    c = -math.sin(k["Parameters"][0])
    k["Evaluation"] = c

def inactiveMax3( k ):
    c = -math.cos(k["Parameters"][1])
    k["Evaluation"] = c

def inactiveMax4( k ):
    c = -math.sin(k["Parameters"][1])
    k["Evaluation"] = c

def stress1( k ):
    c = -k["Parameters"][0] + 6.2
    k["Evaluation"] = c

def stress2( k ):
    c = k["Parameters"][0] - k["Parameters"][1]
    k["Evaluation"] = c

def stress3( k ):
    c = k["Parameters"][0] + 2.0 - 2.0*k["Parameters"][1] 
    k["Evaluation"] = c

def stress4( k ):
    c = 2*k["Parameters"][0] - 3*k["Parameters"][1]
    k["Evaluation"] = c

def stress5( k ):
    c = -(k["Parameters"][0] - 6.28)*(k["Parameters"][1] - 6.28)
    k["Evaluation"] = c

def stress6( k ):
    c = -math.cos(k["Parameters"][0])*math.cos(k["Parameters"][1])
    k["Evaluation"] = c

def stress7( k ):
    c = -math.sin(k["Parameters"][0])*math.sin(k["Parameters"][1])
    k["Evaluation"] = c

def stress8( k ):
    c = k["Parameters"][0] - k["Parameters"][1]**2
    k["Evaluation"] = c

def assert_value( val, expected ):
    assert val  == expected, "Error: Value should be {0} but is {1}\n".format(expected, val)

def assert_smallereq( smaller, greater ):
    assert greater >= smaller, "Error: Value {0} should be smaller {1}\n".format(smaller, greater)

def assert_greatereq( greater, smaller ):
    assert smaller <= greater, "Error: Value {0} should be greater {1}\n".format(greater, smaller)
