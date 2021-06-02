#! /usr/bin/env python3
from subprocess import call
import os

os.environ["OMP_NUM_THREADS"] = "4"
  
r = call(["bash", "get_data.sh"])
if r!=0:
  exit(r)

r = call(["python3", "run-mnist.py", "--test"])
if r!=0:
  exit(r)

exit(0)
