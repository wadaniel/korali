#!/usr/bin/env python3
import os
import sys
import numpy as np

def put_normal_rnds( theta, Ns, fileName ):
  mu  = theta["Parameters"][0]
  var = theta["Parameters"][1]
  
  y = np.random.normal( mu, var, Ns )

  if os.path.isfile( fileName ):
    f = open( fileName, 'a+' )
    np.savetxt( f, np.transpose(y) )
    f.close()

  else:
    sys.exit('put_normal_rnds: file \'{0}\' does not exist! exit..'.format(fileName))
