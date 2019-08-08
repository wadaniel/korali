#!/usr/bin/env python3
import json

def evaluateModel( s ):
   x = s.getVariable(0)
   r = -x*x      
   s.setResult(r)

def assert_value( val, expected ):
    assert val  == expected, "Error: Value should be {0} but is {1}\n".format(expected, val)
