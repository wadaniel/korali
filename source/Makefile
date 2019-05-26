include ../korali.config

GSLPPREFIX = $(shell $(GSLCONFIG) --prefix)
GSLLIBS = $(shell $(GSLCONFIG) --libs)
GSLCFLAGS = $(shell $(GSLCONFIG) --cflags)

CFLAGS = $(shell echo $(CXXFLAGS)) -g -O3 -I. -I../libs $(GSLCFLAGS) -std=c++14 -fPIC 
LDFLAGS = $(SHAREDLIB_FLAG) -Wl,-rpath $(PREFIX)/lib -L$(PREFIX)/lib -L$(GSLPPREFIX)/lib -Wl,-rpath -Wl,$(GSLPPREFIX)/lib $(GSLLIBS) 

CFLAGS += `python3-config --includes` -D_KORALI_USE_PYTHON
LDFLAGS  += `python3-config --ldflags`

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
	@rm -rf *.so *.dylib __init__.py $(OBJECTS)

.PHONY: clean all
