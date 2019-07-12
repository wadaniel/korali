CXXFLAGS = -O3 -g
OBJECTS = jacobi.o

.SECONDARY:
.PHONY: all
all: $(OBJECTS) 

jacobi.o: jacobi.cpp
	$(MPICXX) -c $(CXXFLAGS) $<	

.PHONY: clean
clean:
	$(RM) *.o *.ti *.optrpt *.txt
