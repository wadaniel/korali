all: jacobi.o
CXX=mpic++
KORALICFLAGS=`python3 -m korali.cxx --cflags`

jacobi.o: jacobi.cpp
	$(CXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
