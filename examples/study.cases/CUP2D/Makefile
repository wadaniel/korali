TRANSPORT = run-cmaes-transport run-mocmaes-transport run-vracer-transport 
FISH = run-vracer-fish
WINDMILL = run-vracer-windmill eval-vracer-windmill
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUBISMCFLAGS=$(shell cat _deps/CUP-2D/makefiles/cup.cflags.txt) -I_deps/CUP-2D/source -I_deps/CUP-2D/Cubism/include -DCUBISM
CUBISMLIBS= _deps/CUP-2D/makefiles/libcup.a $(shell cat _deps/CUP-2D/makefiles/cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all
all: transport fish windmill

transport: $(TRANSPORT)

transport2: $(TRANSPORT2)

fish: $(FISH)

windmill: $(WINDMILL)

$(TRANSPORT): %: %.o transportEnvironment.o _deps/CUP-2D/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

$(FISH): %: %.o fishEnvironment.o _deps/CUP-2D/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

$(WINDMILL): %: %.o windmillEnvironment.o _deps/CUP-2D/makefiles/libcup.a 
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
	$(RM) $(TRANSPORT) $(FISH) $(WINDMILL) *.o *.ti *.optrpt *.h5 *.dat *.xmf
