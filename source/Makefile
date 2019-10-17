include Makefile.conf
# ==============================================================================
WARNINGFLAGS = -Wno-deprecated-declarations -Wfatal-errors
MACOS_CXX = clang++
ifeq ($(CXX),$(MACOS_CXX))
   WARNINGFLAGS += -Wno-self-assign-field -Wno-dangling-else
endif

# ==============================================================================
CXXFLAGS  = -g $(OPTFLAGS) -I. $(GSLCFLAGS) $(CXXARCH) $(WARNINGFLAGS)
CXXFLAGS  += $(PYBIND11INCLUDES) $(MPIFLAGS)

LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS)
LDFLAGS += $(PYBIND11LIBS)

DEPFLAGS = -MT $@ -MD -MP -MF $(DEPDIR)/$*.Td

COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) -c -o $@
LINK.o = $(CXX) $(LDFLAGS) $(LDLIBS) -o $@
PRECOMPILE =
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

# ==============================================================================
SOURCES = $(shell find . -name "*.cpp")
BUILDFILES = $(shell find . -name "build.py")

OBJDIR := .o
DEPDIR := .d

OBJS := $(patsubst %,$(OBJDIR)/%.o,$(basename $(SOURCES)))
DEPS := $(patsubst %,$(DEPDIR)/%.d,$(basename $(SOURCES)))

$(shell mkdir -p $(dir $(OBJS)) >/dev/null)
$(shell mkdir -p $(dir $(DEPS)) >/dev/null)
# ==============================================================================

.PHONY: clean all

all: libkorali.so

clean:
	@rm -f `find . -name "*._hpp" | sed 's/._hpp/.hpp/g'`
	@rm -f `find . -name "*._cpp" | sed 's/._cpp/.cpp/g'`
	@rm -f  *.so *.a $(OBJECTS)
	@rm -rf $(OBJDIR) $(DEPDIR)

build:
	@for buildfile in $(BUILDFILES); do python3 $$buildfile; done

libkorali.so: $(OBJS)
	$(LINK.o) $^

$(OBJDIR)/%.o: %.cpp
$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d
	$(PRECOMPILE)
	$(COMPILE.cc) $<
	$(POSTCOMPILE)

.PRECIOUS = $(DEPDIR)/%.d
$(DEPDIR)/%.d: ;

-include $(DEPS)
