from libkorali import *
 
def initialize():
 import shutil
 import os
 if( os.path.exists("./_korali_result/") ) :
  shutil.rmtree("./_korali_result/") 

 return Engine()

