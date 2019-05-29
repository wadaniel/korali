include Makefile.conf

CFLAGS  = -g -O3 -I../include $(GSLCFLAGS) -std=c++14 -Wno-deprecated-declarations -Wno-c++17-extensions
LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS) 

CFLAGS  += $(PYBIND11INCLUDES) -D_KORALI_USE_PYTHON $(MPIFLAGS)
LDFLAGS += $(PYBIND11LIBS)

all: libkorali.so libkorali.a

SOURCES = $(shell find . -name "*.cpp")
OBJECTS_STATIC = $(SOURCES:.cpp=.static.o)
OBJECTS_SHARED = $(SOURCES:.cpp=.shared.o)

%.static.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

%.shared.o: %.cpp
	$(CXX) $(CFLAGS) -fPIC -c $< -o $@

libkorali.so: $(OBJECTS_SHARED)
	$(CXX) $^ -o $@ $(LDFLAGS)

libkorali.a: $(OBJECTS_STATIC)
	ar rs $@ $^

clean:
	@rm -rf *.so *.a $(OBJECTS_STATIC) $(OBJECTS_SHARED)

.PHONY: clean all
