#!/usr/bin/env python
import math

# Minus Eggholder
def m_eggholder(x):
    par1 = x.getParameter(0)
    par2 = x.getParameter(1)

    result = s = -(par2+47.0)*math.sin(math.sqrt(abs(0.5*par1+par2+47.0))) - par1*math.sin(math.sqrt(abs(par1-par2-47.0)));
    
    x.addResult(-result)
