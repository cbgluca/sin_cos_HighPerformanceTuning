all: bench_cos

bench_cos:
	gcc -Wpsabi -Ofast -fopenmp-simd -mavx2 -mfma -march=native -o output/bench_cos cos.c bench_cos.c -lm

bench_sin:
	gcc -Wpsabi -Ofast -fopenmp-simd -mavx2 -mfma -march=native -o output/bench_sin sin.c bench_sin.c -lm
