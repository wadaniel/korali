include Makefile.conf

CFLAGS  = -g $(OPTFLAGS) -I. $(GSLCFLAGS) $(CXXARCH) -Wno-deprecated-declarations -Wfatal-errors
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

.PHONY: clean all

all: libkorali.so

build:
	python3 python/builder/buildKorali.py

SOURCES = $(shell find . -name "*.cpp") 
OBJECTS = $(SOURCES:.cpp=.o)

%.o: %.cpp
	$(CXX) $(CFLAGS) -fPIC -c $< -o $@ -D_KORALI_USE_PYTHON

libkorali.so: $(OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	@rm -f solvers/*/*.hpp solvers/*/*.cpp 
	@rm -f problems/*/*.hpp problems/*/*.cpp 
	@rm -f models/*/*.hpp
	@rm -f distributions/*/*.hpp distributions/*/*.cpp distributions/*.cpp distributions/*.hpp 
	@rm -f variable/*.cpp variable/*.hpp setup.py
	@rm -f setup.py
	@rm -f *.o ./*/*.o *.so *.a $(OBJECTS) 

