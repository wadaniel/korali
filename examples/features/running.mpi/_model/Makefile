all: jacobi.o
KORALICFLAGS=`python3 -m korali.cxx --cflags`

jacobi.o: jacobi.cpp
	$(MPICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
