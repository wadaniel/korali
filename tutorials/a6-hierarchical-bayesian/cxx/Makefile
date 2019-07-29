SOURCES = $(shell ls *.cpp)
BINARIES = $(SOURCES:.cpp=)

KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all data
all: $(BINARIES) data

$(BINARIES) : % : %.o
	$(KORALICXX) -o $@ $^  $(KORALILIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

data:
	./generateData.sh

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt

