

def getcflags(extdir, makeFlags):
 fixedflags='-g -O3 -Wno-deprecated-declarations -I' + extdir
 print(fixedflags + ' ' + makeFlags['OPTFLAGS'] + ' ' + makeFlags['MPIFLAGS'] + ' ' + makeFlags['GSLCFLAGS'] + ' ' + makeFlags['CXXARCH'] + ' ' + makeFlags['PYBIND11INCLUDES'])
