#! /usr/bin/env python3

import json
import matplotlib.pyplot as plt


path = 'korali00000/s00006.json'

with open(path) as f:
    data = json.load(f)
    print(data)

