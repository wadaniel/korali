#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-ffn.py", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "run-rnn.py", "--test"])
if r!=0:
  exit(r)

exit(0)
