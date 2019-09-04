include Makefile.conf

CFLAGS  = -g $(OPTFLAGS) -I. $(GSLCFLAGS) $(CXXARCH) -Wno-deprecated-declarations -Wfatal-errors
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

.PHONY: clean all

all: libkorali.so

BUILDFILES = $(shell find . -name "build.py") 
build:
	@for buildfile in $(BUILDFILES); do python3 $$buildfile; done

SOURCES = $(shell find . -name "*.cpp") 
OBJECTS = $(SOURCES:.cpp=.o)

%.o: %.cpp
	$(CXX) $(CFLAGS) -fPIC -c $< -o $@

libkorali.so: $(OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	@rm -f `find . -name "*._hpp" | sed 's/._hpp/.hpp/g'`
	@rm -f `find . -name "*._cpp" | sed 's/._cpp/.cpp/g'`
	@rm -f *.o ./*/*.o *.so *.a $(OBJECTS) 

