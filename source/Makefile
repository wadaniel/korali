include ../korali.config

CXXFLAGS_SHARED = -fPIC 
CXXFLAGS_STATIC = 

CXXFLAGS = -I. -I../libs -I../libs/gsl/include
CXXFLAGS += -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address -Wno-deprecated-declarations -Wall -Wfatal-errors

ifneq ("$(wildcard ../libs/gsl/lib/libgsl.so)","")
 LIBGSLPATH = "../libs/gsl/lib/libgsl.so" 
endif

ifneq ("$(wildcard ../libs/gsl/lib/libgslcblas.so)","")
 LIBGSLCBLASPATH = "../libs/gsl/lib/libgslcblas.so" 
endif

ifneq ("$(wildcard ../libs/gsl/lib/libgsl.dylib)","")
 LIBGSLPATH = "../libs/gsl/lib/libgsl.dylib" 
endif

ifneq ("$(wildcard ../libs/gsl/lib/libgslcblas.dylib)","")
 LIBGSLCBLASPATH = "../libs/gsl/lib/libgslcblas.dylib" 
endif

LDFLAGS_SHARED = -Wl,--whole-archive $(LIBGSLPATH) $(LIBGSLCBLASPATH) -Wl,--no-whole-archive

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
