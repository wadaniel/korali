

def getcflags(extdir, makeFlags):
 fixedflags='-g -O3 -I' + extdir
 print(fixedflags + ' ' + makeFlags['OPTFLAGS'] + ' ' + makeFlags['MPIFLAGS'] + ' ' + makeFlags['GSLCFLAGS'] + ' ' + makeFlags['CXXARCH'] + ' ' + makeFlags['PYBIND11INCLUDES'])
