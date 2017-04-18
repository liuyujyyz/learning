LNK = $$(pkg-config --cflags --libs python3)

all:
	for f in oprs graph ; do \
		cython $$f.py ; \
	    	gcc $(LNK) -c -fPIC $$f.c -o $$f.o ; \
	    	gcc $$f.o --shared -o $$f.so ; \
	done


