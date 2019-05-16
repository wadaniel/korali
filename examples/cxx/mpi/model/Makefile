CXXFLAGS = -O3 -g -std=c++11 -Wall -Wfatal-errors
OBJECTS = jacobi.o
MPICXX = mpicxx

.SECONDARY:
.PHONY: all
all: $(OBJECTS)

jacobi.o: jacobi.cpp
	$(MPICXX) -c $(CXXFLAGS) $<	

.PHONY: clean
clean:
	$(RM) *.o *.ti *.optrpt *.txt
