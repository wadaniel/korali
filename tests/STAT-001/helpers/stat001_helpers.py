#!/usr/bin/env python3
import json

def open_json(dirname, filename):
    path = '{0}/{1}'.format(dirname, filename)
    data = None
    with open(path) as f:
       data = json.load(f)
    return data
