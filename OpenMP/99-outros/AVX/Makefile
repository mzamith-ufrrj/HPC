################################################################################
#
#  Universidade Federal Rural do Rio de Janeiro - UFRRJ
#  por: Marcelo Zamith - mzamith@ufrrj.br
#
################################################################################
EXEFILE   = exemplo.exec
CPUCC     = g++
CPPFLAGS  =
DEFS      = $(APPLICATION)  $(VERSION)
INCLUDES  = -I.
LIBDIR    = -L/usr/lib 
PAPILIBS  =
LIBS      =  -lm
LINK      =  $(LIBDIR) $(LIBS) -static
MKDIR_P   = mkdir -p
OUT_DIR   = obj

ifeq ($(TARGET), release)
  DEFS      += 
  CPPFLAGS  +=  -O3 
else
  DEFS     += -D_DEBUG
  CPPFLAGS += -g -Wall -O0
endif

CPU_COMPILE = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)


all:	directories loop-explicitly-optimization loop-implicitly-optimization loop-no-optimization main

	$(CPU_COMPILE)	  ${OUT_DIR}/loop-explicitly-optimization.o  ${OUT_DIR}/loop-implicitly-optimization.o  ${OUT_DIR}/loop-no-optimization.o  ${OUT_DIR}/main.o $(LINK) -o $(EXEFILE)

main:
	$(CPU_COMPILE) -c main.cpp -o  ${OUT_DIR}/main.o

loop-no-optimization:
	$(CPU_COMPILE)   -c loop-no-optimization.cpp -o  ${OUT_DIR}/loop-no-optimization.o

loop-implicitly-optimization:
	$(CPU_COMPILE)   -c loop-implicitly-optimization.cpp -o  ${OUT_DIR}/loop-implicitly-optimization.o

loop-explicitly-optimization:
	$(CPU_COMPILE)  -mavx -c loop-explicitly-optimization.cpp -o ${OUT_DIR}/loop-explicitly-optimization.o

directories:
	${MKDIR_P} ${OUT_DIR}

clean:
	rm *.exec; rm obj/*.o

