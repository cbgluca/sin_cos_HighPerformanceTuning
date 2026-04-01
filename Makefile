all:
	gcc -Wpsabi -Ofast -mfma -march=native -o output/programma_hpc testcos.c -lm

sin:
	gcc -Wpsabi -Ofast -mfma -march=native -o output/programma_hpc testsin.c -lm

log:
	gcc -Wpsabi -Ofast -mfma -march=native -o output/programma_hpc log.c -lm