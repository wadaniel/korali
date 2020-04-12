

def getlibs(extdir, makeFlags):
 flags=''
 pythonLibPaths=[ p for p in makeFlags['PYBIND11LIBS'].split() if p.startswith('-L') ]
 for p in pythonLibPaths: flags += ' -Wl,-rpath,' + p.replace('-L', '') + ' '
 flags+='-L' + extdir + ' -lkorali -Wl,-rpath,' + extdir + ' '
 flags+='-L' + makeFlags['GSLPREFIX'] + '/lib -Wl,-rpath,' + makeFlags['GSLPREFIX'] + '/lib '
 flags+='-lpython3'
 print(flags + ' ' + makeFlags['GSLLIBS'] + ' ' + makeFlags['PYBIND11LIBS'])
