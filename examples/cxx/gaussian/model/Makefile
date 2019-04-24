all: gaussian.o

gaussian.o: gaussian.cpp
	$(CXX) -c $< -O3 `gsl-config --cflags`

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
