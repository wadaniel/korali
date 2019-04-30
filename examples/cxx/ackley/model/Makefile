all: ackley.o

ackley: ackley.o
	$(CXX) -o $@ $^ 

ackley.o: ackley.cpp
	$(CXX) -c $< -O3 -g 

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
