include ../korali.config

CXXFLAGS += -g -I. -I../libs -I../libs/gsl/include -std=c++14 -fPIC 
CXXFLAGS += -Wno-unused -Wno-unused-result -Wno-unused-parameter -Wno-address -Wno-deprecated-declarations -Wall -Wfatal-errors -Wno-register -Wno-c++17-extensions

ifneq ("$(wildcard ../libs/gsl/lib/libgsl.so)","")
 LDWHOLE = -Wl,--whole-archive
 LIBGSLPATH = "../libs/gsl/lib/libgsl.so" 
 LIBGSLCBLASPATH = "../libs/gsl/lib/libgslcblas.so" 
 LDNOWHOLE = -Wl,--no-whole-archive
endif

ifneq ("$(wildcard ../libs/gsl/lib/libgsl.dylib)","")
 LDWHOLE = -Wl,-all_load
 LIBGSLPATH = "../libs/gsl/lib/libgsl.dylib" 
 LIBGSLCBLASPATH = "../libs/gsl/lib/libgslcblas.dylib" 
endif

LDFLAGS = $(LDWHOLE) $(LIBGSLPATH) $(LIBGSLCBLASPATH) $(LDNOWHOLE)

CXXFLAGS += `python3-config --includes` -D_KORALI_USE_PYTHON
LDFLAGS  += `python3-config --ldflags`

ifeq ($(USE_MPI), 1)
 CXXFLAGS += -D_KORALI_USE_MPI
 CXX = $(MPICXX)
endif

all: libkorali.so

SOURCES = $(shell find . -name "*.cpp")
OBJECTS = $(SOURCES:.cpp=.o)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS) -c $< -o $@

libkorali.so: $(OBJECTS)
	$(CXX) -shared -O3 $^ -o $@ $(LDFLAGS) 

clean:
	@rm -rf *.so  $(OBJECTS)

.PHONY: clean all
