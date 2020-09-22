BINARIES = run-ddpg
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUBISMLIBS=-L/$(HOME)/CubismUP_2D/makefiles -lcubism -lHYPRE -lfftw3 -lfftw3_omp -fopenmp -lgsl -lgslcblas -L/opt/hdf5/parallel/1.12.0/mpich/3.3/gnu/10.1.0/lib -lhdf5 -L/home/novatig/hypre/build/lib/ 

bs ?= 32
nthreads ?= 24
CUBISMFLAGS= -fopenmp -Wfatal-errors
CUBISMFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -DNTHREADS=$(nthreads)
CUBISMFLAGS+= -I$(HOME)/CubismUP_2D/source -I$(HOME)/CubismUP_2D/Cubism/include
CUBISMFLAGS+= -DNDEBUG -O3 -march=native -mtune=native -DHYPREFFT

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMFLAGS)

model.o: _model/model.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C _model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
