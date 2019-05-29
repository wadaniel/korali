include .Makefile.conf

GSLPPREFIX = $(shell $(GSLCONFIG) --prefix)
GSLLIBS    = $(shell $(GSLCONFIG) --libs)
GSLCFLAGS  = $(shell $(GSLCONFIG) --cflags)

CFLAGS  = -g -O3 -I../include $(GSLCFLAGS) -std=c++14 -fPIC -Wno-deprecated-declarations -Wno-c++17-extensions
LDFLAGS = $(SHAREDLIB_FLAG) -Wl,-rpath $(PREFIX)/lib -L$(PREFIX)/lib -L$(GSLPPREFIX)/lib -Wl,-rpath -Wl,$(GSLPPREFIX)/lib $(GSLLIBS) 

CFLAGS  += `python3 -m pybind11 --includes` -D_KORALI_USE_PYTHON
LDFLAGS += `python3-config --ldflags`

ifeq ($(USE_MPI), 1)
 CFLAGS += -D_KORALI_USE_MPI
 CXX = $(MPICXX)
endif

all: libkorali.so libkorali.a

SOURCES = $(shell find . -name "*.cpp")
OBJECTS = $(SOURCES:.cpp=.o)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

libkorali.so: $(OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS)

libkorali.a: $(OBJECTS)
	ar rs $@ $^

clean:
	@rm -rf *.so *.a *.dylib  $(OBJECTS)

.PHONY: clean all
