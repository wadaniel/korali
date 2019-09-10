#!/usr/bin/env python

# Constraints

def g1(k) :
  c = -127.0 + 2 * k[0] * k[0] + 3.0 * pow(k[1], 4) + k[2] + 4.0 * k[3] * k[3] + 5.0 * k[4]
  k.setResult(c)


def g2(k):
  c = -282.0 + 7.0 * k[0] + 3.0 * k[1] + 10.0 * k[2] * k[2] + k[3] - k[4]
  k.setResult(c)


def g3(k):
  c = -196.0 + 23.0 * k[0] + k[1] * k[1] + 6.0 * k[5] * k[5] - 8.0 * k[6]
  k.setResult(c)

def g4(k):
  c = 4.0 * k[0] * k[0] + k[1] * k[1] - 3.0 * k[0] * k[1] + 2.0 * k[2] * k[2] + 5.0 * k[5] - 11.0 * k[6]
  k.setResult(c)
