BINARIES = main
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o model/mvf.o model/rnd.o suite/testsuite.o suite/enginefactory.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS)

model/mvf.o:
	$(MAKE) -C model all

model/rnd.o:
	$(MAKE) -C model all

suite/testsuite.o:
	$(MAKE) -C suite all

suite/enginefactory.o:
	$(MAKE) -C suite all


%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
