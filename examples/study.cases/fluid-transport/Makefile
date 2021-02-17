BINARIES = run-vracer
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUBISMCFLAGS=$(shell cat cup.cflags.txt) -I_deps/CUP-2D/source -I_deps/CUP-2D/Cubism/include -DCUBISM
CUBISMLIBS= libcup.a $(shell cat cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all 
all: $(BINARIES)

$(BINARIES): % : %.o environment.o libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

environment.o: _environment/environment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.h5 *.dat *.xmf
