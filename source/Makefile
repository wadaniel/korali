include Makefile.conf

CFLAGS  = -g -O3 -I../include $(GSLCFLAGS) -std=c++14 -fPIC -Wno-deprecated-declarations -Wno-c++17-extensions
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) -D_KORALI_USE_PYTHON $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

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
