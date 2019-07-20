SOURCES = $(shell ls *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: clean all

all: libmodel.a

libmodel.a: $(OBJECTS)
	ar rs $@ $^

clean:; rm -rf *.o *.a
