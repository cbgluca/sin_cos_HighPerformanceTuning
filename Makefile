all:
	gcc -Wpsabi -Ofast -mfma -march=native -o output/programma_hpc test.c -lm