#!/usr/bin/env python3
import sys
sys.path.append('./_model')

import json
import argparse

import numpy as np
from agent import *

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='Directory of IRL results.', required=True)

args = parser.parse_args()
print(args)

resfile = f'{args.dir}/latest'
with open(resfile, 'r') as infile:
    results = json.load(infile)
    rawstates = results["Solver"]
    rawactions = obsjson["Actions"]





