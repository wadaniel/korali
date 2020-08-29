all: model.o load_data.o utils.o


curDir := $(shell pwd)
baseKoraliDir=$(curDir)/../../..
include $(baseKoraliDir)/source/Makefile.conf

KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=$(shell python3 -m korali.cxx --cflags)
KORALILIBS=$(shell python3 -m korali.cxx --libs)
KORALICFLAGS += $(GSLCFLAGS)
KORALILIBS += $(GSLLIBS)
## \ that does not seem to work; GSL functions aren't found


# %.o: %.cpp %.hpp load_data.hpp

# test_model: test_model.o model.o load_data.o utils.o
# 	$(KORALICXX) -o $@ $^ $(KORALILIBS)
#
# test_model.o: test_model.cpp test_model.hpp utils.hpp model.hpp load_data.hpp
# 	@echo "Compiling: $< -> $@"
# 	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -c $< -o $@
#
# some_samplers.o: some_samplers.cpp some_samplers.hpp utils.hpp model.hpp
# 	@echo "Compiling: $< -> $@"
# 	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -c $< -o $@


load_data.o: load_data.cpp
	@echo "Compiling: $< -> $@"
	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -g -c $< -o $@


%.o: %.cpp %.hpp utils.hpp
	$(KORALICXX) -c $(KORALICFLAGS) $<


#load_data.o: load_data.cpp  load_data.hpp
	
#	$(KORALICXX) -c $< $(KORALICFLAGS)


model.o: model.cpp model.hpp utils.hpp
	@echo "Compiling: $< -> $@"
	$(KORALICXX) $(CXXFLAGS) $(INCLUDES) $(KORALICFLAGS) -MP -MMD -g -c $< -o $@

#	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
