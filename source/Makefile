include ../korali.config

CXXFLAGS_SHARED = -fPIC 
CXXFLAGS_STATIC = 

CXXFLAGS = -O3 -g -I. -I../libs -I../libs/gsl/include
CXXFLAGS += -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address -Wno-deprecated-declarations -Wall -Wfatal-errors

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
 LDNOWHOLE = -Wl,-noall_load
endif

LDFLAGS_SHARED = $(LDWHOLE) $(LIBGSLPATH) $(LIBGSLCBLASPATH) $(LDNOWHOLE)

ifeq ($(USE_MPI), 1)
endif

ifeq ($(USE_UPCXX), 1)
 CXXFLAGS_STATIC += -D_KORALI_USE_UPCXX `upcxx-meta CPPFLAGS`
endif

ifeq ($(USE_PYTHON), 1)
 CXXFLAGS_SHARED += -D_KORALI_USE_PYTHON `python3 -m pybind11 --includes`
endif

ifeq ($(USE_MULTITHREAD), 1)
 CXXFLAGS += -D_KORALI_USE_MULTITHREAD -pthread
 LDFLAGS_SHARED  += -pthread
endif

all: libkorali.so libkorali.a

SOURCES = $(shell find . -name "*.cpp")
OBJECTS_STATIC = $(SOURCES:.cpp=.static.o)
OBJECTS_SHARED = $(SOURCES:.cpp=.shared.o)

%.static.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_STATIC) -c $< -o $@

%.shared.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_SHARED) -c $< -o $@

libkorali.so: $(OBJECTS_SHARED)
	$(CXX) -shared $^ -o $@ $(LDFLAGS_SHARED) 

libkorali.a: $(OBJECTS_STATIC)
	ar rs $@ $^  

clean:
	@rm -rf *.a *.so $(OBJECTS_STATIC) $(OBJECTS_SHARED)

.PHONY: clean all
