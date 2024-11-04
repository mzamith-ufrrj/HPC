EXEFILE   = GPUInfo.bin
CPUCC     = g++
CPPFLAGS  = -m64 #-m32# -g  -O0 -D_DEBUG
DEFS      =

INCLUDES  = -I.                           \
            -I/usr/include/cuda           \
            -I/usr/local/cuda/include     

LIBDIR   =  -L/usr/lib64                          \
            -L/usr/local/cuda/lib64   

LIBS     =   -lcuda -lcudart

LINK     =  $(LIBDIR) $(LIBS)

C_COMPILE = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)



all:    main
	$(C_COMPILE) main.o  $(LINK) -o $(EXEFILE)

main:
	$(C_COMPILE) -c main.cpp


clean:
	rm *.o; rm $(EXEFILE)
