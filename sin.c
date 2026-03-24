//essenziali
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include<immintrin.h>
#include<time.h>

//aggiunte
#include <string.h>
#include <limits.h>
#include <stdbool.h>

//per confronto
#include <math.h>

#define N 10000000 // 10 milioni di elementi per avere un tempo misurabile

// ==========================================
// QUI METTERAI IL TUO CODICE
// ==========================================
void my_sin_avx(const float *in, float *out, int size)
{
    // Nota: 'size' dovrebbe essere un multiplo di 8 per semplicità,
    // altrimenti devi gestire il "resto" degli elementi alla fine.

    for (int i = 0; i < size; i += 8)
    {
        // 1. Carica 8 float dalla memoria allineata
        __m256 x = _mm256_load_ps(&in[i]);

        // 2. Fai la tua magia matematica qui (Taylor, Range reduction, ecc.)
        // __m256 result = ...

        // ESEMPIO DUMMY (Copia e basta, tu metti il vero calcolo):
        __m256 result = x;

        // 3. Salva gli 8 risultati nella memoria allineata
        _mm256_store_ps(&out[i], result);
    }
}




























// Funzione di utilità per prendere il tempo in secondi
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    // Allocazione allineata a 32-byte (fondamentale per AVX)
    float *in = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_std = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_avx = (float *)_mm_malloc(N * sizeof(float), 32);

    // Inizializza l'array con valori casuali tra -PI e PI
    for (int i = 0; i < N; i++)
    {
        in[i] = ((float)rand() / RAND_MAX) * 2.0f * M_PI - M_PI;
    }

    // --- TEST FUNZIONE STANDARD (Benchmark di base) ---
    double start_std = get_time();
    for (int i = 0; i < N; i++)
    {
        out_std[i] = sinf(in[i]); // sinf è la versione per float a 32 bit
    }
    double end_std = get_time();
    double time_std = end_std - start_std;

    // --- TEST TUA FUNZIONE AVX ---
    double start_avx = get_time();
    my_sin_avx(in, out_avx, N);
    double end_avx = get_time();
    double time_avx = end_avx - start_avx;

    // --- CALCOLO DELL'ERRORE MASSIMO ---
    float max_error = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float error = fabsf(out_std[i] - out_avx[i]);
        if (error > max_error)
        {
            max_error = error;
        }
    }

    // --- STAMPA RISULTATI ---
    printf("Risultati su %d milioni di elementi:\n", N / 1000000);
    printf("Tempo Standard (sinf): %f secondi\n", time_std);
    printf("Tempo Tuo AVX        : %f secondi\n", time_avx);
    printf("Speedup (Acceleraz.): %.2f x\n", time_std / time_avx);
    printf("Errore Massimo       : %e\n", max_error);

    // Pulizia memoria
    _mm_free(in);
    _mm_free(out_std);
    _mm_free(out_avx);

    return 0;
}