all:
	gcc -Wpsabi -Ofast -fopenmp-simd -mavx2 -mfma -march=native -o output/programma_hpc testcos.c -lm

sin:
	gcc -Wpsabi -Ofast -mfma -march=native -o output/programma_hpc testsin.c -lm

log:
	gcc -Wpsabi -Ofast -mfma -march=native -o output/programma_hpc log.c -lm

test:
	gcc -O3 -fno-math-errno -march=native -mfma -o output/programma_hpc testcos.c -lm