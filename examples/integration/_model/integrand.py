#!/usr/bin/env python3
import os
import sys
import numpy as np


def integrand(sample):
  x = sample["Parameters"][0]
  y = sample["Parameters"][1]
  z = sample["Parameters"][2]
  sample["Evaluation"] = x**2 + y**2 + z**2
