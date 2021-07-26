all: jacobi.o
KORALICFLAGS=`python3 -m korali.cxx --cflags`

ifndef MPICXX
	MPICXX=mpic++
endif

jacobi.o: jacobi.cpp
	$(MPICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
