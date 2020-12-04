BINARIES = run-korali
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

ifeq ($(TEST), true)
 CUBISMCFLAGS=-DTEST
 else
 CUBISMCFLAGS=`cat cubism.cflags.txt` -I_deps/cubism/source -I_deps/cubism/Cubism/include -DCUBISM
 CUBISMLIBS=`cat cubism.libs.txt` libcubism.a
endif

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES) : % : %.o environment.o 
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

environment.o: _environment/environment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(MAKE) -C _environment clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.h5 *.dat *.xmf
