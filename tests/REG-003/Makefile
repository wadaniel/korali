SOURCES = $(shell ls *.cpp)
BINARIES = $(SOURCES:.cpp=)

KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model/libmodels.a
	$(KORALICXX) -o $@ $^  $(KORALILIBS) -L./model -lmodels

model/libmodels.a:
	$(MAKE) -C model all

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt

