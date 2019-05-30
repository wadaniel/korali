#!/usr/bin/env python3
import sys
sys.path.append('./model')
from ackley import m_ackley
import korali

k = korali.Engine()
k.loadState('_korali_result/s00001.json')

k.run(m_ackley)
