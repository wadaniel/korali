#!/usr/bin/env python
import math
    
# Minus Levi13
def m_levi13(x):
    par1 = x.getParameter(0)
    par2 = x.getParameter(1)

    result = pow(math.sin(3.0*math.pi*par1),2)+pow((par1-1.0),2)*(1.0+pow(math.sin(3*math.pi*par2),2))+pow(par2-1,2)*(1.0+pow(math.sin(2.0*math.pi*par2),2))
    x.addResult(-result)
