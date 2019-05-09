#!/usr/bin/env python
import math

# Minus Shekel
def m_shekel(x):

    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5 ]
 
    C = [ 4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
          4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6,
          4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
          4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6 ]
 
    result = 0.0
    for i in range(10) :
        for j in range(4) : 
            t = pow( x.getParameter(j) - C[i*4+j], 2) + beta[i]
        result -= 1.0/t

    x.addResult(-result)
