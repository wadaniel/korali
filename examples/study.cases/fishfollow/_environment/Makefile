all: model.o
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`

bs ?= 32
nthreads ?= 24
CUBISMFLAGS= -fopenmp -Wall -Wfatal-errors
CUBISMFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -DNTHREADS=$(nthreads)
CUBISMFLAGS+= -I$(HOME)/CubismUP_2D/source -I$(HOME)/CubismUP_2D/Cubism/include
CUBISMFLAGS+= -DNDEBUG -O3 -march=native -mtune=native

model.o: model.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMFLAGS)

clean:; rm -rf $(TARGET) *.o 

.PHONY: clean all
