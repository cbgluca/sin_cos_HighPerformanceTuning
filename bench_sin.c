/**
 * @file bench_sin.c
 * @brief High-performance benchmark comparing standard math library sine with
 *        AVX2 Minimax vectorized and 4x unrolled implementations from sin.c.
 *
 * Benchmark Methodology:
 * 1. Cache-Resident Working Set:
 *    An array of 4096 single-precision floats occupies exactly 16 KB (4096 * 4 bytes).
 *    This fits completely within the CPU's L1 Data Cache (typically 32 KB to 48 KB per core),
 *    eliminating DRAM memory latency bottlenecks and isolating raw computational throughput.
 *
 * 2. High Iteration Count:
 *    Running 250,000 iterations over 4,096 elements results in 1,024,000,000 (1.024 Billion)
 *    total sine evaluations, providing stable, statistically reliable measurements.
 *
 * 3. Numerical Verification:
 *    Before timing, each implementation is validated against standard library sinf() to verify
 *    maximum absolute error and accuracy bounds.
 *
 * 4. Anti-DCE (Dead Code Elimination) Barrier:
 *    Compilers with -O3 or -Ofast could eliminate calculations whose results are never read.
 *    A dummy checksum accumulator ensures that all output values are strictly consumed.
 */

#include "sin.h"

/**
 * @brief Get high-resolution monotonic time in seconds.
 * @return Monotonic time as a double-precision float.
 */
static inline double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void)
{
    // =========================================================================
    // CONFIGURATION & SETUP
    // =========================================================================
    const int N = 4096;               // 4096 floats * 4 bytes = 16 KB (fits in L1D cache)
    const int iterations = 250000;    // 250k passes -> ~1.024 Billion total float calculations
    const double total_ops = (double)N * (double)iterations;

    printf("======================================================================\n");
    printf("               AVX2 SINE HIGH-PERFORMANCE BENCHMARK                   \n");
    printf("======================================================================\n");
    printf("Array Size      : %d elements (%.1f KB)\n", N, (N * sizeof(float)) / 1024.0f);
    printf("Iterations      : %d\n", iterations);
    printf("Total Operations: %.2e floats\n", total_ops);
    printf("CPU Cache Target: L1 Data Cache (~32-48 KB)\n");
    printf("----------------------------------------------------------------------\n\n");

    // Allocate 32-byte aligned memory buffers to satisfy AVX2 aligned load/store requirements
    float *in            = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_std       = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_minimax   = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_unrolled  = (float *)_mm_malloc(N * sizeof(float), 32);

    if (!in || !out_std || !out_minimax || !out_unrolled)
    {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Seed random number generator and initialize input array in range [0, 2*PI]
    srand(42);
    for (int i = 0; i < N; i++)
    {
        in[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * (float)M_PI;
    }

    // =========================================================================
    // NUMERICAL ACCURACY VERIFICATION
    // =========================================================================
    // Run single pass to verify correctness and compute maximum error vs math.h
    for (int i = 0; i < N; i++)
    {
        out_std[i] = sinf(in[i]);
    }
    my_sin_avx_minimax(in, out_minimax, N);
    my_sin_avx_minimax_unrolled(in, out_unrolled, N);

    float max_err_minimax  = 0.0f;
    float max_err_unrolled = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float err_m = fabsf(out_std[i] - out_minimax[i]);
        float err_u = fabsf(out_std[i] - out_unrolled[i]);
        if (err_m > max_err_minimax)  max_err_minimax  = err_m;
        if (err_u > max_err_unrolled) max_err_unrolled = err_u;
    }

    printf("--- ACCURACY VERIFICATION (vs standard sinf) ---\n");
    printf("Max Error (Minimax)         : %.6e\n", max_err_minimax);
    printf("Max Error (Minimax Unrolled): %.6e\n", max_err_unrolled);
    printf("----------------------------------------------------------------------\n\n");

    // =========================================================================
    // BENCHMARK 1: Standard math.h (sinf - scalar / auto-vectorized baseline)
    // =========================================================================
    double start_std = get_time();
    for (int k = 0; k < iterations; k++)
    {
        for (int i = 0; i < N; i++)
        {
            out_std[i] = sinf(in[i]);
        }
    }
    double time_std = get_time() - start_std;
    double thr_std  = total_ops / time_std;
    double lat_std  = (time_std / total_ops) * 1e9;

    // =========================================================================
    // BENCHMARK 2: AVX2 Minimax (my_sin_avx_minimax)
    // =========================================================================
    double start_minimax = get_time();
    for (int k = 0; k < iterations; k++)
    {
        my_sin_avx_minimax(in, out_minimax, N);
    }
    double time_minimax = get_time() - start_minimax;
    double thr_minimax  = total_ops / time_minimax;
    double lat_minimax  = (time_minimax / total_ops) * 1e9;

    // =========================================================================
    // BENCHMARK 3: AVX2 Minimax 4x Unrolled (my_sin_avx_minimax_unrolled)
    // =========================================================================
    double start_unrolled = get_time();
    for (int k = 0; k < iterations; k++)
    {
        my_sin_avx_minimax_unrolled(in, out_unrolled, N);
    }
    double time_unrolled = get_time() - start_unrolled;
    double thr_unrolled  = total_ops / time_unrolled;
    double lat_unrolled  = (time_unrolled / total_ops) * 1e9;

    // =========================================================================
    // ANTI-DEAD-CODE-ELIMINATION (DCE) BARRIER
    // =========================================================================
    // Accumulate elements to prevent compiler from optimizing away memory writes
    float checksum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        checksum += out_std[i] + out_minimax[i] + out_unrolled[i];
    }

    // =========================================================================
    // RESULTS REPORT
    // =========================================================================
    printf("\n======================================================================\n");
    printf("                           BENCHMARK RESULTS                          \n");
    printf("======================================================================\n");

    printf("[1] Standard math.h (sinf):\n");
    printf("    Total Execution Time : %8.4f seconds\n", time_std);
    printf("    Throughput           : %8.2e operations/sec\n", thr_std);
    printf("    Average Latency      : %8.2f ns/op\n", lat_std);
    printf("    Speedup              :    1.00x (Baseline)\n\n");

    printf("[2] AVX2 Minimax (my_sin_avx_minimax):\n");
    printf("    Total Execution Time : %8.4f seconds\n", time_minimax);
    printf("    Throughput           : %8.2e operations/sec\n", thr_minimax);
    printf("    Average Latency      : %8.2f ns/op\n", lat_minimax);
    printf("    Speedup              : %8.2fx vs baseline\n\n", time_std / time_minimax);

    printf("[3] AVX2 Minimax Unrolled (my_sin_avx_minimax_unrolled):\n");
    printf("    Total Execution Time : %8.4f seconds\n", time_unrolled);
    printf("    Throughput           : %8.2e operations/sec\n", thr_unrolled);
    printf("    Average Latency      : %8.2f ns/op\n", lat_unrolled);
    printf("    Speedup              : %8.2fx vs baseline\n\n", time_std / time_unrolled);

    printf("======================================================================\n");
    if (checksum == 12345.67f)
    {
        // Never triggered, but references checksum so it cannot be eliminated
        printf("Checksum: %f\n", checksum);
    }

    // Free 32-byte aligned buffers
    _mm_free(in);
    _mm_free(out_std);
    _mm_free(out_minimax);
    _mm_free(out_unrolled);

    return EXIT_SUCCESS;
}
