#!/usr/bin/env python3
import math

def estimateHarvest(ph, temp):
 b = getBananaCount(ph, temp)
 o = getOrangeCount(ph, temp)
 a = getAppleCount(ph, temp)
 totalHarvestQuality = b*1.2 + o*0.9 + a*1.1
   
 return totalHarvestQuality, b, o, a 

def getBananaCount(ph, temp):
 a = 100
 b = ((ph - 7.3))**2 + ((temp - 35)*0.1)**2
 c = 2
 t = a * math.exp(-b / c)
 return math.floor(t)
 
def getOrangeCount(ph, temp):
 a = 50
 b = ((ph - 6.6))**2 + ((temp - 27)*0.1)**2
 c = 2
 t = a * math.exp(-b / c)
 return math.floor(t)
 
def getAppleCount(ph, temp):
 a = 90
 b = ((ph - 8.6))**2 + ((temp - 20)*0.1)**2
 c = 2
 t = a * math.exp(-b / c)
 return math.floor(t)
