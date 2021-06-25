#!/usr/bin/env python3
import math


def evaluateModel(s):
  x1 = s["Parameters"][0]
  x2 = s["Parameters"][1]
  r = -x1**2 - x2**2 - math.sin(x1)**2 - math.sin(x2)**2
  s["F(x)"] = r


def inactive1(k):
  k["F(x)"] = -1


def inactive2(k):
  k["F(x)"] = -2


def activeMax1(k):
  c = -(k["Parameters"][0] - 1.0)
  k["F(x)"] = c


def activeMax2(k):
  c = -(k["Parameters"][0] - 2.0)
  k["F(x)"] = c


def activeMax3(k):
  c = -(k["Parameters"][1] - 1.0)
  k["F(x)"] = c


def activeMax4(k):
  c = -(k["Parameters"][1] - 2.0)
  k["F(x)"] = c


def inactiveMax1(k):
  c = -math.cos(k["Parameters"][0])
  k["F(x)"] = c


def inactiveMax2(k):
  c = -math.sin(k["Parameters"][0])
  k["F(x)"] = c


def inactiveMax3(k):
  c = -math.cos(k["Parameters"][1])
  k["F(x)"] = c


def inactiveMax4(k):
  c = -math.sin(k["Parameters"][1])
  k["F(x)"] = c


def stress1(k):
  c = -k["Parameters"][0] + 6.2
  k["F(x)"] = c


def stress2(k):
  c = k["Parameters"][0] - k["Parameters"][1]
  k["F(x)"] = c


def stress3(k):
  c = k["Parameters"][0] + 2.0 - 2.0 * k["Parameters"][1]
  k["F(x)"] = c


def stress4(k):
  c = 2 * k["Parameters"][0] - 3 * k["Parameters"][1]
  k["F(x)"] = c


def stress5(k):
  c = -(k["Parameters"][0] - 6.28) * (k["Parameters"][1] - 6.28)
  k["F(x)"] = c


def stress6(k):
  c = -math.cos(k["Parameters"][0]) * math.cos(k["Parameters"][1])
  k["F(x)"] = c


def stress7(k):
  c = -math.sin(k["Parameters"][0]) * math.sin(k["Parameters"][1])
  k["F(x)"] = c


def stress8(k):
  c = k["Parameters"][0] - k["Parameters"][1]**2
  k["F(x)"] = c


def assert_value(val, expected):
  assert val == expected, "Error: Value should be {0} but is {1}\n".format(
      expected, val)


def assert_smallereq(smaller, greater):
  assert greater >= smaller, "Error: Value {0} should be smaller {1}\n".format(
      smaller, greater)


def assert_greatereq(greater, smaller):
  assert smaller <= greater, "Error: Value {0} should be greater {1}\n".format(
      greater, smaller)
