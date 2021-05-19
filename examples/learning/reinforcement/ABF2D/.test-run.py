#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-vracer.py", "--test"])
if r!=0:
  exit(r)

exit(0)
