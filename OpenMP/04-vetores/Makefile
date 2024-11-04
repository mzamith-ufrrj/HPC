CPUCC       = g++
CPPFLAGS    = -g  -fopenmp #-O3 #-fopenmp
DEFS        = -DALING=64
INCLUDES    = -I.
LIBDIR      =  #-L/opt/gnu-gcc/lib64
LIBS        = -lm #-lgomp

LINK        = $(LIBDIR) $(LIBS)
COMPILECPU  = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)

all: 00-main-hello 01-main-hello-vet 02-main-vet-add-vets  03-main-vet-find

00-main-hello:
	$(COMPILECPU) 00-main-hello.cpp $(LINK)  -o hello-omp.exec

01-main-hello-vet:
	$(COMPILECPU) 01-main-hello-vet.cpp $(LINK)  -o hello-vet-omp.exec

02-main-vet-add-vets:
	$(COMPILECPU) 02-main-vet-add-vets.cpp $(LINK)  -o hello-vet-add-omp.exec


03-main-vet-find:
	$(COMPILECPU) 03-main-vet-find.cpp $(LINK)  -o hello-vet-find-omp.exec



clean:
	rm  rm *.exec
