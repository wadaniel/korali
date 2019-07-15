#!/usr/bin/env python3
import json

def open_json(dirname, filename):
    path = '{0}/{1}'.format(dirname, filename)
    data = None
    with open(path) as f:
       data = json.load(f)
    return data

def assert_value( fields, src1, src2 ):
    for field in fields:
        src1 = src1.get(field)
        src2 = src2.get(field)

    assert src1 == src2, "Error: Values of field {0} differ ({1} vs {2})\n".format(field, src1, src2)
 

def assert_boolean( field, val ):
    assert field.getBoolean() == val

def assert_string( field, val ):
    assert field.getString() == val
