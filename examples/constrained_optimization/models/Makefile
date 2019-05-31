all: models.a
SOURCES=$(shell find . -name "*.cpp")
OBJECTS=$(SOURCES:.cpp=.o)

models.a: $(OBJECTS)
	ar rs $@ $^

clean:; rm -rf $(TARGET) *.o *.a

.PHONY: clean all
