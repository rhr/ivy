CC	= gcc
CFLAGS	= -O3 -fPIC
FC	= gfortran
FFLAGS	= -O3 -fPIC

LIBS	= -llapack -lblas
FOBJS	= expokit.o mataid.o clock.o
COBJS	= dexpm_c.o
CYMOD	= cyexpokit.so

all:	$(CYMOD) $(FOBJS) $(COBJS)

clean: 
	rm *.o

.f.o:
	$(FC) $(FFLAGS) -c $<

%.o : %.f90
	$(CC) $(CFLAGS) -c $<

$(CYMOD): cyexpokit.pyx $(FOBJS) $(COBJS)
	rm -f cyexpokit.c
	python ./setup.py build_ext --inplace
