CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
OBJECTS = jacobi.o
MPICXX = mpicxx

.SECONDARY:
.PHONY: all
all: $(OBJECTS) main

main: jacobi.o main.o
	$(MPICXX) -o $@ $?

jacobi.o: jacobi.cpp
	$(MPICXX) -c $(CXXFLAGS) $<	

main.o: main.cpp
	$(MPICXX) -c $(CXXFLAGS) $<	

.PHONY: clean
clean:
	$(RM) *.o *.ti *.optrpt *.txt main
