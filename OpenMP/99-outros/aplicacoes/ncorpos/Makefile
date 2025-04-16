CPUCC       = gcc #/opt/gnu-gcc/bin/gcc
CPUC++      = g++ #/opt/gnu-gcc/bin/g++
CUDACC      = nvcc
CPPFLAGS    = -O3 -fopenmp
CUDAFLAGS   = -O3 
DEFS        = -DALING=64
INCLUDES    = -I.
LIBDIR      =      #-L/opt/gnu-gcc/lib64
LIBS        = -lm #-lcuda

LINK        = $(LIBDIR) $(LIBS)
COMPILE     = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)
COMPILE++   = $(CPUC++) $(DEFS) $(INCLUDES) $(CPPFLAGS)
COMPILEGPU  = $(CUDACC) $(DEFS) $(INCLUDES) $(CUDAFLAGS)

#all:   n-bodies-1.0 n-bodies-2.0 n-bodies-2.1 n-bodies-2.2 memory
all:   n-bodies-1.0 n-bodies-2.0 n-bodies-cuda memory

n-bodies-1.0:
	$(COMPILE) -c n-bodies-1.0.c -o n-bodies-1.0.o
	$(COMPILE)  n-bodies-1.0.o    $(LINK) -o n-bodies-1.0.exec

n-bodies-2.0:
	$(COMPILE) -c n-bodies-2.0.c -o n-bodies-2.0.o
	$(COMPILE)  n-bodies-2.0.o    $(LINK) -o n-bodies-2.0.exec

n-bodies-cuda:
	$(COMPILEGPU) -c n-bodies.cu -o n-bodies.o
	$(COMPILEGPU)  n-bodies.o    $(LINK) -o n-bodies-cuda.exec



n-bodies-2.1:
	$(COMPILE) -c n-bodies-2.1.c -o n-bodies-2.1.o
	$(COMPILE)  n-bodies-2.1.o    $(LINK) -o n-bodies-2.1.exec

n-bodies-2.2:
	$(COMPILE) -c n-bodies-2.2.c -o n-bodies-2.2.o
	$(COMPILE)  n-bodies-2.2.o    $(LINK) -o n-bodies-2.2.exec

memory:
	$(COMPILE++) -c memory.cpp -o memory.o
	$(COMPILE++)  memory.o    $(LINK) -o memory.exec

clean:
	rm *.o; rm *.exec
