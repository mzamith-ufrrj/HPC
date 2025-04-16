CPUC++      = g++ #/opt/gnu-gcc/bin/g++
CPPFLAGS    = -O0 -fopenmp #-O3 #-fopenmp
DEFS        = -DALING=64
INCLUDES    = -I.
LIBDIR      =      #-L/opt/gnu-gcc/lib64
LIBS        = -lm
MKDIR_P     = mkdir -p
OUT_DIR     = obj
LINK        = $(LIBDIR) $(LIBS)
COMPILE++   = $(CPUC++) $(DEFS) $(INCLUDES) $(CPPFLAGS)

all:  directories gol-cpu main-gol

main-gol:
	$(COMPILE++) -c main-gol.cpp -o obj/main-gol.o
	$(COMPILE++)  obj/gol-cpu.o obj/main-gol.o    $(LINK) -o GOL-Benchmarking.exec

gol-cpu:
	$(COMPILE++) -c gol-cpu.cpp -o obj/gol-cpu.o

directories:
	${MKDIR_P} ${OUT_DIR}
clean:
	rm obj/*.o; rm *.exec
