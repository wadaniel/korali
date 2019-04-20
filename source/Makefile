include ../common.mk
CXXFLAGS += -fPIC
#CXXFLAGS += $(shell python3 -m pybind11 --includes)
CXXFLAGS += -I.
CXXFLAGS += -I../libs

SOURCES = $(shell find . -name "*.cpp")
OBJECTS = $(SOURCES:.cpp=.o)

all: libkorali.so

libkorali.so: $(OBJECTS) 
	$(CXX) -shared $^ -o $@ -Wl,-rpath=$(HOME)/usr/lib -lgsl -lgslcblas -L$(HOME)/usr/lib

clean:; rm -rf *.a *.o $(OBJECTS)

.PHONY: clean all
