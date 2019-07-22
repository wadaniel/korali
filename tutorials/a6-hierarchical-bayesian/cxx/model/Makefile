SOURCES = $(shell ls *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.PHONY: clean all

all: libmodel.a

libmodel.a: $(OBJECTS)
	ar rs $@ $^

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf *.o *.a
