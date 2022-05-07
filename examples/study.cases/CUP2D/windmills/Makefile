WINDMILL = run-vracer-windmill eval-vracer-windmill 
OTHERWINDMILL = other-run-vracer-windmill # eval-vracer-windmill
KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUPFOLDER=../_deps/CUP-2D
CUBISMCFLAGS=$(shell cat ${CUPFOLDER}/makefiles/cup.cflags.txt) -I${CUPFOLDER}/source -I${CUPFOLDER}/Cubism/include -DCUBISM -Wno-shadow
CUBISMLIBS= ${CUPFOLDER}/makefiles/libcup.a $(shell cat ${CUPFOLDER}/makefiles/cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all
all: windmill otherwindmill

windmill: $(WINDMILL)
otherwindmill: $(OTHERWINDMILL)

$(WINDMILL): %: %.o windmillEnvironment.o ${CUPFOLDER}/makefiles/libcup.a 
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

$(OTHERWINDMILL): %: %.o otherwindmillEnvironment.o ${CUPFOLDER}/makefiles/libcup.a 
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

windmillEnvironment.o: _model/windmillEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

otherwindmillEnvironment.o: _model/otherwindmillEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(RM) $(WINDMILL) $(OTHERWINDMILL) *.o *.ti *.optrpt *.h5 *.xmf
