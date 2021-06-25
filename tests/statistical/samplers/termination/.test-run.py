#! /usr/bin/env python3
from subprocess import call

tmcmc_criteria = [
  "Max Generations",
  "Max Generations",
  "Target Annealing Exponent"
  ]

tmcmc_values = [
  1,     # Max Generations
  3,     # Max Generations
  0.6    # Target Annealing Exponent
  ]

for c, v in zip(tmcmc_criteria, tmcmc_values):
  cmd = ["python3", "tmcmc_termination.py", "--criterion", f'{c}', "--value", f"{v}"]
  r = call( cmd )
  if r!=0:
    exit(r)


nested_criteria = [
  "Max Generations",
  "Max Effective Sample Size"
  ]

nested_values = [
  10,    # Max Generations
  100    # Max Effective Sample Size
  ]

for c, v in zip(nested_criteria, nested_values):
  cmd = ["python3", "nested_termination.py", "--criterion", f'{c}', "--value", f"{v}"]
  r = call( cmd )
  if r!=0:
    exit(r)


exit(0)
