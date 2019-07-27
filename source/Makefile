include Makefile.conf

CFLAGS  = -g $(OPTFLAGS) -I. $(GSLCFLAGS) $(CXXARCH) -Wno-deprecated-declarations -Wfatal-errors
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

.PHONY: clean all

all: solvers/config.cpp libkorali.so libkorali.a

solvers/config.cpp:
	python3 ./buildKorali.py

SOURCES = $(shell find . -name "*.cpp") solvers/config.cpp
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
	@rm -rf *.so *.a solvers/config.cpp solvers/*/*.hpp $(OBJECTS_STATIC) $(OBJECTS_SHARED)

