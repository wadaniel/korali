include ../Makefile.conf

PREFIX = $(HOME)
SED    = sed

TARGET = korali.pc

CXXFLAGS  = -g $(OPTFLAGS) $(GSLCFLAGS) $(CXXARCH) $(WARNINGFLAGS)
CXXFLAGS += $(PYBIND11INCLUDES) $(MPIFLAGS)

LDFLAGS = $(SHAREDLIB_FLAG) -L$(GSLPREFIX)/lib -Wl,-rpath -Wl,$(GSLPREFIX)/lib $(GSLLIBS)
LDFLAGS += $(PYBIND11LIBS)

INSTALL_DIR = "$(PREFIX)/share/pkgconfig"

install: $(TARGET)
	@mkdir -p -- $(INSTALL_DIR)
	@cp $^ $(INSTALL_DIR)
	@echo "Installed $^ in $(INSTALL_DIR)"


korali.pc: korali.pc.in
	@$(SED) "s;@CFLAGS@;$(CXXFLAGS);g; s;@LIBS@;$(LDFLAGS);g" $< > $@

clean:; rm -f $(TARGET)

.PHONY: install clean
