BINARIES = run-saem  run-saem-gaussian-mixture
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o model/model.o  model/load_data.o  model/some_samplers.o
	$(KORALICXX) -o $@ $^ $(KORALILIBS)

model/%.o:
	$(MAKE) -C model all

%.o: %.cpp
	$(KORALICXX) -c $(KORALICFLAGS) $<

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt
