all: ackley.o

$(TARGET).o: $(TARGET).cpp
	$(CXX) -c $< -O3 -g 

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
