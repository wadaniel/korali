include Makefile.conf

CFLAGS  = -g $(OPTFLAGS) -I../include $(GSLCFLAGS) $(CXXARCH) -Wno-deprecated-declarations
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

all: libkorali.so libkorali.a

SOURCES = $(shell find . -name "*.cpp") config.cpp
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

config.cpp:
	python3 ../python/parser/build_config.py

clean:
	@rm -rf *.so *.a config.cpp $(OBJECTS_STATIC) $(OBJECTS_SHARED)

.PHONY: clean all
