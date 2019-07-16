#!/usr/bin/env python3

def assert_value( field, val ):
    assert field.getValue() == val, "Error: Value should be {0} but is {1}\n".format(val, field.getValue())
 

def assert_boolean( field, val ):
    assert field.getBoolean() == val

def assert_string( field, val ):
    assert field.getString() == val


def evaluateModel( s ):
   x = s.getVariable(0)
   r = -x*x      
   s.addResult(r)

def g1(x):
    return -3.14


