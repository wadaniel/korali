#!/usr/bin/env python3
import os
import sys
import numpy as np

def integrand( sample ):
  x  = sample["Parameters"][0] 
  y= x**2
  sample["Evaluation"] = y
