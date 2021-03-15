TRANSPORT = run-cmaes-transport run-mocmaes-transport run-vracer-transport 
FISH = run-vracer-fish
WINDMILL = run-vracer-windmill
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUBISMCFLAGS=$(shell cat cup.cflags.txt) -I_deps/CUP-2D/source -I_deps/CUP-2D/Cubism/include -DCUBISM
CUBISMLIBS= libcup.a $(shell cat cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all
all: transport fish windmill

transport: $(TRANSPORT)

fish: $(FISH)

windmill: $(WINDMILL)

$(TRANSPORT): %: %.o transportEnvironment.o libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

$(FISH): %: %.o fishEnvironment.o libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

$(WINDMILL): %: %.o windmillEnvironment.o libcup.a 
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

fishEnvironment.o: _models/fishEnvironment/fishEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

transportEnvironment.o: _models/transportEnvironment/transportEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

windmillEnvironment.o: _models/windmillEnvironment/windmillEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(RM) $(TRANSPORT) $(FISH) *.o *.ti *.optrpt *.h5 *.dat *.xmf
