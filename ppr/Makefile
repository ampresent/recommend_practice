CFLAGS = -m64 -Wall -Wextra -Wformat=2 -std=gnu90 -O2 -march=native\
		 -L/usr/lib/python2.7 -lpython2.7\
		 -I/usr/include/python2.7
SPQRLIBS = -lspqr -lsuitesparseconfig -lcholmod -lamd -lcolamd -lm -llapack -lblas
spqr_wrapper.so: spqr_wrapper.c
	gcc $^ -o $@ $(CFLAGS) -fPIC -shared $(SPQRLIBS)
clean:
	rm -f spqr_wrapper.so
