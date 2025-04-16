CPUCC       = g++
CPPFLAGS    = -O2 -fopenmp #-O3 #-fopenmp
DEFS        = -DALING=64
INCLUDES    = -I.
LIBDIR      =  #-L/opt/gnu-gcc/lib64
LIBS        = -lm

LINK        = $(LIBDIR) $(LIBS)
COMPILECPU  = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)

all:
	$(COMPILECPU) main-trapezios.cpp $(LINK)  -o trapezios.exec

clean:
	rm *.exec
