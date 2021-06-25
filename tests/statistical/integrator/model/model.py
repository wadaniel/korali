#!/usr/bin/env python3
import os
import sys
import numpy as np

def model_integration(s):
  x = s["Parameters"][0]
  s["Evaluation"] = x**2
