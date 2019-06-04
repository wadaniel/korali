CXXFLAGS = -O3 -g
OBJECTS = jacobi.o
KORALICXX=$(shell python3 -m korali.cxx --compiler)

.SECONDARY:
.PHONY: all
all: $(OBJECTS) main

main: jacobi.o main.o
	$(KORALICXX) -o $@ $?

jacobi.o: jacobi.cpp
	$(KORALICXX) -c $(CXXFLAGS) $<	

main.o: main.cpp
	$(KORALICXX) -c $(CXXFLAGS) $<	

.PHONY: clean
clean:
	$(RM) *.o *.ti *.optrpt *.txt main
