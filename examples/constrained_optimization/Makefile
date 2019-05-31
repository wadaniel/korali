BINARIES = maximize_constrained_rosenbrock maximize_constrained_g09 maximize_unconstrained_g09 maximize_constrained_tr2
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o models/models.a
	$(KORALICXX) -o $@ $^  $(KORALILIBS)

models/models.a:
	$(MAKE) -C models all

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C models clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
