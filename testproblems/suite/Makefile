KORALICXX=$(shell python3 -m korali.cxx --compiler)                             
KORALICFLAGS=`python3 -m korali.cxx --cflags`                                   

all: testsuite.o

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o *.a

.PHONY: clean all

