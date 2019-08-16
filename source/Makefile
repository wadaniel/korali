include Makefile.conf

CFLAGS  = -g $(OPTFLAGS) -I. $(GSLCFLAGS) $(CXXARCH) -Wno-deprecated-declarations -Wfatal-errors
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

.PHONY: clean all

all: libkorali.so

build:
	python3 python/builder/buildAll.py

SOURCES = $(shell find . -name "*.cpp") 
OBJECTS = $(SOURCES:.cpp=.o)

%.o: %.cpp
	$(CXX) $(CFLAGS) -fPIC -c $< -o $@

libkorali.so: $(OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	@rm -f solvers/*/*.hpp solvers/*/*.cpp 
	@rm -f conduits/*/*.hpp conduits/*/*.cpp 
	@rm -f problems/*/*.hpp problems/*/*.cpp 
	@rm -f models/*/*.hpp
	@rm -f distributions/univariate/*/*.hpp distributions/univariate/*/*.cpp distributions/univariate/*.cpp distributions/univariate/*.hpp 
	@rm -f variable/*.cpp variable/*.hpp setup.py
	@rm -f setup.py
	@rm -f *.o ./*/*.o *.so *.a $(OBJECTS) 

