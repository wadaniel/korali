all: model.o
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`

bs ?= 32
nthreads ?= 24
CPPFLAGS= -fopenmp -Wall
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -DNTHREADS=$(nthreads)
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow
CPPFLAGS+= -Wwrite-strings -Wno-unused-parameter #-Wno-cpp
CPPFLAGS+= -Wno-float-equal
CPPFLAGS+= -Wno-redundant-decls
CPPFLAGS+= -I$(HOME)/CubismUP_2D/source -I$(HOME)/CubismUP_2D/Cubism/include
CPPFLAGS+= -DNDEBUG -O3 -march=native -mtune=native

model.o: model.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CPPFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
