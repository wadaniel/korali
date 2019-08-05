#!/usr/bin/env python

# Constraints

def g1(k) :
  x = k.getVariables();
  c = -127.0 + 2 * x[0] * x[0] + 3.0 * pow(x[1], 4) + x[2] + 4.0 * x[3] * x[3] + 5.0 * x[4];
  k.setConstraintEvaluation(c);


def g2(k):
  x = k.getVariables();
  c = -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] * x[2] + x[3] - x[4];
  k.setConstraintEvaluation(c);


def g3(k):
  x = k.getVariables();
  c = -196.0 + 23.0 * x[0] + x[1] * x[1] + 6.0 * x[5] * x[5] - 8.0 * x[6];
  k.setConstraintEvaluation(c);

def g4(k):
  x = k.getVariables();
  c = 4.0 * x[0] * x[0] + x[1] * x[1] - 3.0 * x[0] * x[1] + 2.0 * x[2] * x[2] + 5.0 * x[5] - 11.0 * x[6];
  k.setConstraintEvaluation(c);
