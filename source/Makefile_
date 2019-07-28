include Makefile.conf

CFLAGS  = -g $(OPTFLAGS) -I. $(GSLCFLAGS) $(CXXARCH) -Wno-deprecated-declarations -Wfatal-errors
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

.PHONY: clean all

all: libkorali.so libkorali.a

build:
	python3 python/builder/buildKorali.py

SOURCES = $(shell find . -name "*.cpp") 
OBJECTS_STATIC = $(SOURCES:.cpp=.static.o)
OBJECTS_SHARED = $(SOURCES:.cpp=.shared.o)

%.static.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ 

%.shared.o: %.cpp
	$(CXX) $(CFLAGS) -fPIC -c $< -o $@ -D_KORALI_USE_PYTHON

libkorali.so: $(OBJECTS_SHARED)
	$(CXX) $^ -o $@ $(LDFLAGS)

libkorali.a: $(OBJECTS_STATIC)
	ar rs $@ $^

clean:
	@rm -rf *.so *.a solvers/*/*.hpp solvers/*/*.cpp 
	@rm -rf *.so *.a variable.cpp variable.hpp setup.py
	@rm -rf *.so *.a $(OBJECTS_STATIC) $(OBJECTS_SHARED)

