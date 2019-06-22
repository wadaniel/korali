KORALICXX=$(shell python3 -m korali.cxx --compiler)                             
KORALICFLAGS=`python3 -m korali.cxx --cflags`                                   
KORALILIBS=`python3 -m korali.cxx --libs`     

all: testsuite.o
	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o *.a

.PHONY: clean all

