#!/usr/bin/env python3

def evaluateModel( s ):
   r = -s[0]*s[0]      
   s.setResult(r)

def assert_value( val, expected ):
    assert val  == expected, "Error: Value should be {0} but is {1}\n".format(expected, val)

def assert_smallereq( smaller, greater ):
    assert greater >= smaller, "Error: Value {0} should be smaller {1}\n".format(smaller, greater)

def assert_greatereq( greater, smaller ):
    assert smaller <= greater, "Error: Value {0} should be greater {1}\n".format(greater, smaller)
