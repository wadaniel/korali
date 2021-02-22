BINARIES = run-cmaes-transport run-vracer-transport run-vracer-fish
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUBISMCFLAGS=$(shell cat cup.cflags.txt) -I_deps/CUP-2D/source -I_deps/CUP-2D/Cubism/include -DCUBISM
CUBISMLIBS= libcup.a $(shell cat cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all
all: $(BINARIES)

run-cmaes-transport: run-cmaes-transport.o transportEnvironment.o libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

run-vracer-transport: run-vracer-transport.o transportEnvironment.o libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

run-vracer-fish: run-vracer-fish.o fishEnvironment.o libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

run-cmaes-transport.o: run-cmaes-transport.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

run-vracer-transport.o: run-vracer-transport.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

run-vracer-fish.o: run-vracer-fish.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

fishEnvironment.o: _fishEnvironment/fishEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

transportEnvironment.o: _transportEnvironment/transportEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.h5 *.dat *.xmf
