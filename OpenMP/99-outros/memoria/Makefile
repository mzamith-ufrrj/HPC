CPUC++      = g++ #/opt/gnu-gcc/bin/g++
CPPFLAGS    = -O3
DEFS        =
INCLUDES    = -I.  -I$(PAPI)/testlib -I$(PAPI)/include -I$(PAPI)/share/papi/testlib
LIBDIR      =
LIBS        = -lm
PAPI      = /opt/papi
LINK        = $(LIBDIR) $(LIBS) -static
COMPILE++   = $(CPUC++) $(DEFS) $(INCLUDES) $(CPPFLAGS)
MKDIR_P   = mkdir -p
OUT_DIR   = obj


all:  directories memory

memory:
	$(COMPILE++) -c memory.cpp -o ${OUT_DIR}/memory.o
	$(COMPILE++)  ${OUT_DIR}/memory.o    $(LINK) -o memory.exec  $(PAPI)/lib/libpapi.a

directories:
	${MKDIR_P} ${OUT_DIR}
clean:
	rm ${OUT_DIR}/*.o; rm *.exec
