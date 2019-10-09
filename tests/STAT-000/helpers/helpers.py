#!/usr/bin/env python3

def assert_value( field, val ):
    assert field == val, "Error: Value should be {0} but is {1}\n".format(val, field)
 

def assert_boolean( field, val ):
    assert field == val

def assert_string( field, val ):
    assert field == val


def evaluateModel( s ):
   x = s["Parameters"][0]
   r = -x*x      
   s["Evaluation"] = r

def g1(x):
    return -3.14


