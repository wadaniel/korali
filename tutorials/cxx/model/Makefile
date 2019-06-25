SOURCES = $(shell ls *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: clean all

all: libmodels.a

libmodels.a: $(OBJECTS)
	ar rs $@ $^

clean:; rm -rf *.o *.a
