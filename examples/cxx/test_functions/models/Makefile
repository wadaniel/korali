all: models.a
SOURCES=$(shell find . -name "*.cpp")
OBJECTS=$(SOURCES:.cpp=.o)

models.a: $(OBJECTS)
	ar rs $@ $^

CXXFLAGS = -O3 -g -std=c++11

clean:; rm -rf $(TARGET) *.o *.a

.PHONY: clean all
