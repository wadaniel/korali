include ../korali.config

KORALI_LIBNAME_SHARED=libkorali.so
KORALI_LIBNAME_STATIC=libkorali.a

GSL_LIBNAME=../libs/gsl/lib/libgsl.$(DLL_EXTENSION)
GSLCBLAS_LIBNAME=../libs/gsl/lib/libgslcblas.$(DLL_EXTENSION)

CFLAGS = $(shell echo $(CXXFLAGS)) -g -O3 -I. -I../libs -I../libs/gsl/include -std=c++14 -fPIC 
LDFLAGS = $(SHAREDLIB_FLAG) $(GSL_LIBNAME) $(GSLCBLAS_LIBNAME)

CFLAGS += `python3-config --includes` -D_KORALI_USE_PYTHON
LDFLAGS  += `python3-config --ldflags`

ifeq ($(USE_MPI), 1)
 CFLAGS += -D_KORALI_USE_MPI
 CXX = $(MPICXX)
endif

all: $(KORALI_LIBNAME_SHARED) $(KORALI_LIBNAME_STATIC)

SOURCES = $(shell find . -name "*.cpp")
OBJECTS = $(SOURCES:.cpp=.o)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

$(KORALI_LIBNAME_SHARED): $(OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS)
	
$(KORALI_LIBNAME_STATIC): $(OBJECTS)
	ar rs $@ $^  
	
clean:
	@rm -rf *.so *.dylib $(OBJECTS)

.PHONY: clean all
