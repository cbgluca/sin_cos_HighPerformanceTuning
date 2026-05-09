#include "lib.h"

double my_cos(double);
void my_cos_avx(const float*, float*, int);
void my_cos_avx_minimax(const float*, float *, int);
void my_cos_avx_minimax_unrolled(const float *, float *, int);
void my_cos_avx_minimax_cody(const float *, float *, int);
void avx_cos_minimax_array(const float *, float *, size_t);
void my_cos_avx_speed_only(const float *, float *, int);
void rust_cos_avx_port(const float *, float *, int);
void rust_cos_avx_port_optimized(const float *, float *, int);

void bench_custom_avx_pd(size_t);
__m256d _mm256_cos_pd(__m256d);
void bench_my_avx_minimax_ps(size_t);
static inline __m256 _mm256_cos_minimax_ps(__m256);
void my_cos_avx_minimax_zero2pi(const float *, float *, int);

    double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main()
{
    // SETUP: 4096 float = 16 KB. Entra perfettamente nella Cache L1D (solitamente 32-48 KB)
    const int N = 4096;
    const int iterazioni = 250000;                            // Un quarto di milione di passaggi per durare ~1 secondo
    const double ops_totali = (double)N * (double)iterazioni; // ~1 Miliardo di float calcolati

    printf("=== BENCHMARK ALGORITMICO IN CACHE L1 ===\n");
    printf("Array Size: %d elementi (16 KB)\n", N);
    printf("Iterazioni: %d\n", iterazioni);
    printf("Volume Tot: %.0f float processati\n\n", ops_totali);

    float *in = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_std = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_my = (float *)_mm_malloc(N * sizeof(float), 32);
    float *out_my2 = (float *)_mm_malloc(N * sizeof(float), 32);

    // Inizializza input nel range [0, 2PI]
    for (int i = 0; i < N; i++)
    {
        in[i] = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
    }

    // --- TEST 1: Baseline (math.h) ---
    double start_std = get_time();
    for (int k = 0; k < iterazioni; k++)
    {
        for (int i = 0; i < N; i++)
        {
            out_std[i] = cosf(in[i]);
        }
    }
    double time_std = get_time() - start_std;
    double thr_std = ops_totali / time_std;
    double lat_std = (time_std / ops_totali) * 1e9;

    // --- TEST 2: Tua Implementazione Minimax ---
    double start_my = get_time();
    for (int k = 0; k < iterazioni; k++)
    {
        rust_cos_avx_port(in, out_my, N);
    }
    double time_my = get_time() - start_my;
    double thr_my = ops_totali / time_my;
    double lat_my = (time_my / ops_totali) * 1e9;

    // --- TEST 3: Tua Implementazione Minimax ---
    double start_my2 = get_time();
    for (int k = 0; k < iterazioni; k++)
    {
        rust_cos_avx_port_optimized(in, out_my2, N);
    }
    double time_my2 = get_time() - start_my2;
    double thr_my2 = ops_totali / time_my2;
    double lat_my2 = (time_my2 / ops_totali) * 1e9;

    // --- BARRIERA ANTI-DCE ---
    // Somma i risultati per costringere il compilatore a non tagliare l'esecuzione
    float sum_dummy = 0.0f;
    for (int i = 0; i < N; i++)
        sum_dummy += out_my[i] + out_std[i];

    // --- RISULTATI ---
    printf("[1] math.h (Scalare/Auto-Vec)\n");
    printf("    Tempo totale : %f s\n", time_std);
    printf("    Throughput   : %.2e ops/sec\n", thr_std);
    printf("    Latenza media: %.2f ns/op\n\n", lat_std);

    printf("[2] my_cos_avx_minimax_unrolled\n");
    printf("    Tempo totale : %f s\n", time_my);
    printf("    Throughput   : %.2e ops/sec\n", thr_my);
    printf("    Latenza media: %.2f ns/op\n\n", lat_my);

    printf("[3] not\n");
    printf("    Tempo totale : %f s\n", time_my2);
    printf("    Throughput   : %.2e ops/sec\n", thr_my2);
    printf("    Latenza media: %.2f ns/op\n\n", lat_my2);

    printf(">>> VANTAGGIO ARCHITETTURALE: %.2fx <<<\n\n", time_std / time_my);
    printf(">>> VANTAGGIO ARCHITETTURALE2: %.2fx <<<\n\n", time_std / time_my2);

    // Evita che il compilatore rimuova sum_dummy
    if (sum_dummy == 0.0f)
        printf(".\n");

    _mm_free(in);
    _mm_free(out_std);
    _mm_free(out_my);
    return 0;
}

// int main(){
    
//     // printf("Insert value for sin: ");
//     //double x;
//     // scanf("%lf", &x);
//     // double result = sinf(x);
//     // printf("real sin(%lf) = %lf\n", x, result);


//     // double result_approx = my_sin(x);
//     // printf("approximate sin(%lf) = %lf\n", x, result_approx);

//     int N = 100000000; // 10000000    4096
//     int iterazioni = 100000;


//     printf("Generazione di %d valori casuali in corso...\n", N);

//     // Alloco memoria per gli array
//     float *in = _mm_malloc(N * sizeof(float), 32);
//     float *out_std = _mm_malloc(N * sizeof(float), 32);
//     float *out_my = _mm_malloc(N * sizeof(float), 32);
//     float *out_my2 = _mm_malloc(N * sizeof(float), 32);

//     // // random tra -1mln e 1mln
//     // for (int i = 0; i < N; i++) {
//     //     // 1. Genera un numero tra 0.0f e 1.0f
//     //     float frazione = (float)rand() / (float)RAND_MAX; 
        
//     //     // 2. Moltiplicalo per un range gigantesco (es. 2 milioni)
//     //     float ampiezza = 2000000.0f;
        
//     //     // 3. Spostalo per avere sia positivi che negativi (da -1 milione a +1 milione)
//     //     in[i] = (frazione * ampiezza) - (ampiezza / 2.0f);
//     // }

//     //between +10pi e -10pi
//     for (int i = 0; i < N; i++) {
//         in[i] = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
//     }

//     // // --- TEST LIBRERIA STANDARD ---
//     // double start_std = get_time();

//     // for (int k = 0; k < iterazioni; k++)
//     // {
//     // #pragma omp simd
//     //     for (int i = 0; i < N; i++)
//     //     {
//     //         out_std[i] = cosf(in[i]); // cosf, NON cos
//     //     }
//     // }
//     // double end_std = get_time();

//     // double time_std = end_std - start_std;

//     //--- OLD STANDARD ---
//     double start_std = get_time();

//     for (int i = 0; i < N; i++)
//     {
//         out_std[i] = cosf(in[i]); // cosf, NON cos  
//     }
//     double end_std = get_time();

//     double time_std = end_std - start_std;

//     // // --- TEST TUA FUNZIONE ---
//     // double start_my = get_time();

//     // for (int k = 0; k < iterazioni; k++)
//     // {
//     //     my_cos_avx_minimax(in, out_my, N);
//     // }

//     // double end_my = get_time();
//     // double time_my = end_my - start_my;

//     // //test seconda funz
//     // double start_my2 = get_time();

//     // for (int k = 0; k < iterazioni; k++)
//     // {
//     //     my_cos_avx_minimax_unrolled(in, out_my, N);
//     // }

//     // double end_my2 = get_time();
//     // double time_my2 = end_my2 - start_my2;

//     // --- OLD TEST ---
//     double start_my = get_time();
    
//     my_cos_avx_minimax_zero2pi(in, out_my, N);
    
//     double end_my = get_time();
//     double time_my = end_my - start_my;



//     // test seconda funz
//     double start_my2 = get_time();


//     rust_cos_avx_port(in, out_my2, N);
 

//     double end_my2 = get_time();
//     double time_my2 = end_my2 - start_my2;

//     // --- CALCOLO ERRORE MASSIMO ---
//     double max_error = 0.0;
//     double max_error2 = 0.0;
//     for (int i = 0; i < N; i++)
//     {
//         double error = fabs(out_std[i] - out_my[i]);
//         double error2 = fabs(out_std[i] - out_my2[i]);
//         if (error > max_error)
//         {
//             max_error = error;
//         }
//         if(error2 > max_error2){
//             max_error2 = error2;
//         }

//     }

//     // --- STAMPA RISULTATI ---
//     printf("\n=== RISULTATI BENCHMARK ===\n");
//     printf("Tempo Standard (math.h): %f secondi\n", time_std);
//     printf("Tempo my_cos_roll           : %f secondi\n", time_my);
//     printf("Tempo my_cos_minmax           : %f secondi\n", time_my2);

//     if (time_my < time_std) {
//         printf("Performance            : La tua è più VELOCE di %.2fx!\n", time_std / time_my);
//     } else {
//         printf("Performance            : La tua è più LENTA di %.2fx\n", time_my / time_std);
//     }

//     if (time_my2 < time_std)
//     {
//         printf("Performance            : La tua è più VELOCE di %.2fx!\n", time_std / time_my2);
//     }
//     else
//     {
//         printf("Performance            : La tua è più LENTA di %.2fx\n", time_my2 / time_std);
//     }

//     printf("Errore Massimo         : %e   errore 2: %e\n", max_error, max_error2);



//     // printf("\n=== AVVIO BENCHMARK RUST PORTATO ===\n");

//     // // Passa 100 milioni come numero totale di operazioni
//     // size_t ops_totali = 100000000;
//     // bench_custom_avx_pd(ops_totali);

//     // N = 4096;
//     // iterazioni = 100000;

//     // printf("Inizio benchmark comparativo (N=%d, Iterazioni=%d)\n", N, iterazioni);

//     // // 2. Allocazione memoria allineata
//     // float *inpu = _mm_malloc(N * sizeof(float), 32);
//     // float *outu_std = _mm_malloc(N * sizeof(float), 32);
//     // float *outu_rust = _mm_malloc(N * sizeof(float), 32);
//     // float *outu_minimax = _mm_malloc(N * sizeof(float), 32);

//     // // 3. Inizializzazione dati
//     // for (int i = 0; i < N; i++)
//     // {
//     //     in[i] = ((float)rand() / RAND_MAX) * 4.0f * M_PI - 2.0f * M_PI;
//     // }

//     // // 4. TEST LIBRERIA STANDARD (Baseline)
//     // double startu_std = get_time();
//     // for (int k = 0; k < iterazioni; k++)
//     // {
//     //     for (int i = 0; i < N; i++)
//     //     {
//     //         out_std[i] = cosf(in[i]);
//     //     }
//     // }
//     // double timeu_std = get_time() - startu_std;

//     // // 5. TEST PORTING RUST (Taylor grado 19)
//     // double startu_rust = get_time();
//     // for (int k = 0; k < iterazioni; k++)
//     // {
//     //     rust_cos_avx_port(in, outu_rust, N);
//     // }
//     // double timeu_rust = get_time() - startu_rust;

//     // // 6. TEST TUA MINIMAX (Chebyshev grado 4)
//     // double start_minimax = get_time();
//     // for (int k = 0; k < iterazioni; k++)
//     // {
//     //     my_cos_avx_minimax(in, outu_minimax, N);
//     // }
//     // double time_minimax = get_time() - start_minimax;

//     // // 7. CALCOLO ERRORI MASSIMI
//     // float err_rust = 0, err_minimax = 0;
//     // for (int i = 0; i < N; i++)
//     // {
//     //     float e_r = fabsf(out_std[i] - outu_rust[i]);
//     //     float e_m = fabsf(out_std[i] - outu_minimax[i]);
//     //     if (e_r > err_rust)
//     //         err_rust = e_r;
//     //     if (e_m > err_minimax)
//     //         err_minimax = e_m;
//     // }

//     // // 8. STAMPA CONFRONTO
//     // printf("\n=== RISULTATI CONFRONTO ===\n");
//     // printf("Tempo math.h     : %f s\n", time_std);
//     // printf("Tempo Rust Port  : %f s (Speedup: %.2fx)\n", timeu_rust, time_std / timeu_rust);
//     // printf("Tempo Tua Minimax: %f s (Speedup: %.2fx)\n", time_minimax, time_std / time_minimax);

//     // printf("\n=== PRECISIONE ===\n");
//     // printf("Max Errore Rust   : %e\n", err_rust);
//     // printf("Max Errore Minimax: %e\n", err_minimax);

//     // _mm_free(inpu);
//     // _mm_free(outu_std);
//     // _mm_free(outu_rust);
//     // _mm_free(outu_minimax);

//     // size_t ops2_totali = 100000000;

//     // printf("=== AVVIO BENCHMARK COMPARATIVO ===\n");
//     // printf("Target operazioni: %zu\n\n", ops2_totali);

//     // printf("--- Test 1: Rust Port (Double Precision / Taylor Grado 19) ---\n");
//     // bench_custom_avx_pd(ops2_totali);

//     // printf("\n--- Test 2: Tua Minimax (Single Precision / Grado 4) ---\n");
//     // bench_my_avx_minimax_ps(ops2_totali);

//     return 0;
// }

double my_cos(double x){
    //bringing it ot the right interval [-pi, +pi]
    double test, x_2,x_4;

    const double inv_pi = 1.0 / M_PI;

    int k = round(x * inv_pi);

    x = x - (k * M_PI);

    x_2= x*x;
    x_4= x_2*x_2;
    //x_6 = x_4*x_2;
    //x_8 = x_6 * x_2;

    // 1 - x_2 * (1.0/2.0) + x_4 * (1.0/24.0) + x_6 * (1.0/720.0) + x_8 * (1.0/40320.0);
    test = 1.0 - x_2 * (1.0/2.0) + x_4 * (1.0/24.0);

    return (k & 1) ? -test : test; // if odd cos(x+pi) = -cos(x)
}


void my_cos_avx(const float* in, float* out, int size){

    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi     = _mm256_set1_ps((float)M_PI);

    float c_2 = 1.0f / 2.0f;
    float c_4 = 1.0f / 24.0f;
    __m256 vec_c2 = _mm256_set1_ps(c_2);
    __m256 vec_c4 = _mm256_set1_ps(c_4);
    __m256 vec_1 = _mm256_set1_ps(1.0f);

    int i;
    
    for (i=0; i <= size - 8; i += 8) {
        __m256 vec_x = _mm256_load_ps(&in[i]);

        // x * inv_pi
        __m256 x_inv_pi = _mm256_mul_ps(vec_x, vec_inv_pi);

        //__MM_FROUND_TO_NEAREST_INT when rounding to the nearest int it remains a __m256
        __m256 v_k = _mm256_round_ps(x_inv_pi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // (k * M_PI)
        __m256 k_pi = _mm256_mul_ps(v_k, vec_pi);

        // x = x - k_pi
        vec_x = _mm256_sub_ps(vec_x, k_pi);


        __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);
        __m256 vec_x4 = _mm256_mul_ps(vec_x2, vec_x2);
        //__m256 vec_x6 = _mm256_mul_ps(vec_x4, vec_x2);

        __m256 vec_result = _mm256_fnmadd_ps(vec_x2, vec_c2, vec_1);
        vec_result = _mm256_fmadd_ps(vec_x4, vec_c4, vec_result);
        //__m256 vec_term3 = _mm256_mul_ps(vec_x2, vec_c2);
        //__m256 vec_term5 = _mm256_mul_ps(vec_x4, vec_c4);

        //__m256 vec_res_parziale = _mm256_sub_ps(vec_x, vec_term3);
        //__m256 vec_result       = _mm256_add_ps(vec_res_parziale, vec_term5);

        __m256i k_int = _mm256_cvtps_epi32(v_k);
        __m256i sign_mask = _mm256_slli_epi32(k_int, 31);
        vec_result = _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask));

        _mm256_store_ps(&out[i], vec_result);
    }
    //for all the remaining floats that don't fit in a YMM, i use the my_cos() one by one
    for (; i < size; i++) {
        out[i] = my_cos(in[i]); 
    }
    
}

void my_cos_avx_minimax(const float *in, float *out, int size)
{

    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);

    __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    int i;

    for (i = 0; i <= size - 8; i += 8)
    {
        __m256 vec_x = _mm256_load_ps(&in[i]);

        // x * inv_pi
        __m256 x_inv_pi = _mm256_mul_ps(vec_x, vec_inv_pi);

        //__MM_FROUND_TO_NEAREST_INT when rounding to the nearest int it remains a __m256
        __m256 v_k = _mm256_round_ps(x_inv_pi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // (k * M_PI)
        __m256 k_pi = _mm256_mul_ps(v_k, vec_pi);

        // x = x - k_pi
        vec_x = _mm256_sub_ps(vec_x, k_pi);

        // x is now in [-pi/2 , +pi/2]
        // chebychev coefficients



        __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);

        // Calcolo di P(x) = c_0 + c_2 * x^2 + c_4 * x^4
        // HORNER METHOD to skip 1 operation: P(x) = c_0 + x^2 * (c_2 + x^2 * c_4)

        __m256 vec_result = _mm256_fmadd_ps(vec_x2, vec_c4, vec_c2);
        vec_result = _mm256_fmadd_ps(vec_result, vec_x2, vec_c0);

        __m256i k_int = _mm256_cvtps_epi32(v_k);                                // converts from float to epi32
        __m256i sign_mask = _mm256_slli_epi32(k_int, 31);                       // left shit, to get sign
        vec_result = _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask)); // cats is a bypass to make xor work. tricks xor to think its a float and not a epi32

        _mm256_store_ps(&out[i], vec_result);
    }
    // for all the remaining floats that don't fit in a YMM, i use the my_cos() one by one
    for (; i < size; i++)
    {
        out[i] = my_cos(in[i]);
    }
}

void my_cos_avx_minimax_unrolled(const float *in, float *out, int size)
{
    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);
    __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    int i;

    // 2. Loop Unrolling x4: 32 float per iteration
    for (i = 0; i <= size - 32; i += 32)
    {
        __m256 x0 = _mm256_load_ps(&in[i]);
        __m256 x1 = _mm256_load_ps(&in[i + 8]);
        __m256 x2 = _mm256_load_ps(&in[i + 16]);
        __m256 x3 = _mm256_load_ps(&in[i + 24]);

        // k = round(x * inv_pi)
        __m256 k0 = _mm256_round_ps(_mm256_mul_ps(x0, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k1 = _mm256_round_ps(_mm256_mul_ps(x1, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k2 = _mm256_round_ps(_mm256_mul_ps(x2, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k3 = _mm256_round_ps(_mm256_mul_ps(x3, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        x0 = _mm256_fnmadd_ps(k0, vec_pi, x0);
        x1 = _mm256_fnmadd_ps(k1, vec_pi, x1);
        x2 = _mm256_fnmadd_ps(k2, vec_pi, x2);
        x3 = _mm256_fnmadd_ps(k3, vec_pi, x3);

        // x^2
        __m256 x0_2 = _mm256_mul_ps(x0, x0);
        __m256 x1_2 = _mm256_mul_ps(x1, x1);
        __m256 x2_2 = _mm256_mul_ps(x2, x2);
        __m256 x3_2 = _mm256_mul_ps(x3, x3);

        // Polinomio Minimax: c_0 + x^2 * (c_2 + x^2 * c_4)
        __m256 res0 = _mm256_fmadd_ps(x0_2, vec_c4, vec_c2);
        __m256 res1 = _mm256_fmadd_ps(x1_2, vec_c4, vec_c2);
        __m256 res2 = _mm256_fmadd_ps(x2_2, vec_c4, vec_c2);
        __m256 res3 = _mm256_fmadd_ps(x3_2, vec_c4, vec_c2);

        res0 = _mm256_fmadd_ps(res0, x0_2, vec_c0);
        res1 = _mm256_fmadd_ps(res1, x1_2, vec_c0);
        res2 = _mm256_fmadd_ps(res2, x2_2, vec_c0);
        res3 = _mm256_fmadd_ps(res3, x3_2, vec_c0);

        // sign mask
        __m256i sign_mask0 = _mm256_slli_epi32(_mm256_cvtps_epi32(k0), 31);
        __m256i sign_mask1 = _mm256_slli_epi32(_mm256_cvtps_epi32(k1), 31);
        __m256i sign_mask2 = _mm256_slli_epi32(_mm256_cvtps_epi32(k2), 31);
        __m256i sign_mask3 = _mm256_slli_epi32(_mm256_cvtps_epi32(k3), 31);

        res0 = _mm256_xor_ps(res0, _mm256_castsi256_ps(sign_mask0));
        res1 = _mm256_xor_ps(res1, _mm256_castsi256_ps(sign_mask1));
        res2 = _mm256_xor_ps(res2, _mm256_castsi256_ps(sign_mask2));
        res3 = _mm256_xor_ps(res3, _mm256_castsi256_ps(sign_mask3));

        // Store
        _mm256_store_ps(&out[i], res0);
        _mm256_store_ps(&out[i + 8], res1);
        _mm256_store_ps(&out[i + 16], res2);
        _mm256_store_ps(&out[i + 24], res3);
    }

    // for those that don't fill the array completely in the end
    for (; i < size; i++)
    {
        out[i] = my_cos(in[i]);
    }
}

void my_cos_avx_minimax_cody(const float *in, float *out, int size)
{

    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);

    __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    // La somma di pi_1 + pi_2 + pi_3 fa M_PI, ma applicati in sequenza proteggono i bit bassi
    __m256 vec_pi_1 = _mm256_set1_ps(3.140625f); // Parte alta (0x40490f00)
    __m256 vec_pi_2 = _mm256_set1_ps(0.00096702575f);    // Parte media (0x3a7da000)
    __m256 vec_pi_3 = _mm256_set1_ps(6.277114e-07f);     // Parte bassa (0x35285c50)

    int i;

    for (i = 0; i <= size - 8; i += 8)
    {
        __m256 vec_x = _mm256_load_ps(&in[i]);


        __m256 x_inv_pi = _mm256_mul_ps(vec_x, vec_inv_pi);
        __m256i k_int = _mm256_cvtps_epi32(x_inv_pi); // La conversione arrotonda già all'intero più vicino
        __m256 v_k = _mm256_cvtepi32_ps(k_int);       // Riconverte in float per la moltiplicazione

        // Riduzione Cody-Waite a catena
        // x = x - (k * pi_1)
        vec_x = _mm256_fnmadd_ps(v_k, vec_pi_1, vec_x);
        // x = x - (k * pi_2)
        vec_x = _mm256_fnmadd_ps(v_k, vec_pi_2, vec_x);
        // x = x - (k * pi_3)
        vec_x = _mm256_fnmadd_ps(v_k, vec_pi_3, vec_x);

        // x is now in [-pi/2 , +pi/2]
        // chebychev coefficients

        __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);

        // Calcolo di P(x) = c_0 + c_2 * x^2 + c_4 * x^4
        // HORNER METHOD to skip 1 operation: P(x) = c_0 + x^2 * (c_2 + x^2 * c_4)

        __m256 vec_result = _mm256_fmadd_ps(vec_x2, vec_c4, vec_c2);
        vec_result = _mm256_fmadd_ps(vec_result, vec_x2, vec_c0);


        __m256i sign_mask = _mm256_slli_epi32(k_int, 31);                       // left shit, to get sign
        vec_result = _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask)); // cats is a bypass to make xor work. tricks xor to think its a float and not a epi32

        _mm256_store_ps(&out[i], vec_result);
    }
    // for all the remaining floats that don't fit in a YMM, i use the my_cos() one by one
    for (; i < size; i++)
    {
        out[i] = my_cos(in[i]);
    }
}



// Compilare con: gcc -Ofast -march=native
void avx_cos_minimax_array(const float *in, float *out, size_t size)
{
    __m256 inv_pi = _mm256_set1_ps(0.318309886f); // 1 / PI
    __m256 pi = _mm256_set1_ps(3.141592654f);

    // Costanti del polinomio Minimax ottimizzato per [-PI/2, PI/2] (precisione ~10^-5)
    __m256 c0 = _mm256_set1_ps(0.999993f);
    __m256 c1 = _mm256_set1_ps(-0.499912f);
    __m256 c2 = _mm256_set1_ps(0.041487f);
    __m256 c3 = _mm256_set1_ps(-0.001271f);

    for (size_t i = 0; i < size; i += 8)
    {
        __m256 x = _mm256_loadu_ps(&in[i]);

        // 1. Riduzione del dominio a [-PI/2, PI/2]
        // Troviamo l'intero multiplo di PI più vicino a x
        __m256 y = _mm256_mul_ps(x, inv_pi);
        __m256 round_y = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // x_rem = x - round_y * PI (tramite FMA: -(round_y * PI) + x)
        __m256 x_rem = _mm256_fnmadd_ps(round_y, pi, x);

        // 2. Calcolo del segno tramite manipolazione bit a bit
        // Se round_y è un numero dispari, il coseno cambia segno.
        __m256i y_int = _mm256_cvtps_epi32(round_y);
        // Estrai l'ultimo bit (1 se dispari, 0 se pari) e shiftalo sul bit di segno del float (31)
        __m256i sign_bit = _mm256_slli_epi32(_mm256_and_si256(y_int, _mm256_set1_epi32(1)), 31);

        // 3. Valutazione del polinomio di Horner con FMA: c0 + x^2 * (c1 + x^2 * (c2 + c3 * x^2))
        __m256 x2 = _mm256_mul_ps(x_rem, x_rem);
        __m256 res = _mm256_fmadd_ps(c3, x2, c2);
        res = _mm256_fmadd_ps(res, x2, c1);
        res = _mm256_fmadd_ps(res, x2, c0);

        // 4. Iniezione del segno tramite XOR logico
        res = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(res), sign_bit));

        _mm256_storeu_ps(&out[i], res);
    }
}

void my_cos_avx_speed_only(const float *in, float *out, int size)
{
    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);

    // Coefficienti Grado 4 Minimax
    __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    int i;
    for (i = 0; i <= size - 32; i += 32)
    {
        // Prefetch dei dati che serviranno tra 2 iterazioni (64 float avanti)
        _mm_prefetch((const char *)&in[i + 64], _MM_HINT_T0);
        _mm_prefetch((const char *)&in[i + 80], _MM_HINT_T0);

        __m256 x0 = _mm256_load_ps(&in[i]);
        __m256 x1 = _mm256_load_ps(&in[i + 8]);
        __m256 x2 = _mm256_load_ps(&in[i + 16]);
        __m256 x3 = _mm256_load_ps(&in[i + 24]);

        // Calcolo k e arrotondamento fusi in una singola conversione per risparmiare latenza
        __m256i k0_int = _mm256_cvtps_epi32(_mm256_mul_ps(x0, vec_inv_pi));
        __m256i k1_int = _mm256_cvtps_epi32(_mm256_mul_ps(x1, vec_inv_pi));
        __m256i k2_int = _mm256_cvtps_epi32(_mm256_mul_ps(x2, vec_inv_pi));
        __m256i k3_int = _mm256_cvtps_epi32(_mm256_mul_ps(x3, vec_inv_pi));

        __m256 k0_float = _mm256_cvtepi32_ps(k0_int);
        __m256 k1_float = _mm256_cvtepi32_ps(k1_int);
        __m256 k2_float = _mm256_cvtepi32_ps(k2_int);
        __m256 k3_float = _mm256_cvtepi32_ps(k3_int);

        // Range reduction sporca (1 FMA)
        x0 = _mm256_fnmadd_ps(k0_float, vec_pi, x0);
        x1 = _mm256_fnmadd_ps(k1_float, vec_pi, x1);
        x2 = _mm256_fnmadd_ps(k2_float, vec_pi, x2);
        x3 = _mm256_fnmadd_ps(k3_float, vec_pi, x3);

        __m256 x0_2 = _mm256_mul_ps(x0, x0);
        __m256 x1_2 = _mm256_mul_ps(x1, x1);
        __m256 x2_2 = _mm256_mul_ps(x2, x2);
        __m256 x3_2 = _mm256_mul_ps(x3, x3);

        // Polinomio di Horner (solo grado 4)
        __m256 res0 = _mm256_fmadd_ps(x0_2, vec_c4, vec_c2);
        __m256 res1 = _mm256_fmadd_ps(x1_2, vec_c4, vec_c2);
        __m256 res2 = _mm256_fmadd_ps(x2_2, vec_c4, vec_c2);
        __m256 res3 = _mm256_fmadd_ps(x3_2, vec_c4, vec_c2);

        res0 = _mm256_fmadd_ps(res0, x0_2, vec_c0);
        res1 = _mm256_fmadd_ps(res1, x1_2, vec_c0);
        res2 = _mm256_fmadd_ps(res2, x2_2, vec_c0);
        res3 = _mm256_fmadd_ps(res3, x3_2, vec_c0);

        // Maschere di segno
        res0 = _mm256_xor_ps(res0, _mm256_castsi256_ps(_mm256_slli_epi32(k0_int, 31)));
        res1 = _mm256_xor_ps(res1, _mm256_castsi256_ps(_mm256_slli_epi32(k1_int, 31)));
        res2 = _mm256_xor_ps(res2, _mm256_castsi256_ps(_mm256_slli_epi32(k2_int, 31)));
        res3 = _mm256_xor_ps(res3, _mm256_castsi256_ps(_mm256_slli_epi32(k3_int, 31)));

        // STREAMING STORES: Scrittura diretta in RAM
        _mm256_stream_ps(&out[i], res0);
        _mm256_stream_ps(&out[i + 8], res1);
        _mm256_stream_ps(&out[i + 16], res2);
        _mm256_stream_ps(&out[i + 24], res3);
    }

    // Le istruzioni non-temporal richiedono una barriera di memoria alla fine
    _mm_sfence();

    for (; i < size; i++)
    {
        out[i] = my_cos(in[i]);
    }
}

void rust_cos_avx_port(const float *in, float *out, int size)
{
    __m256 vec_pi_2 = _mm256_set1_ps((float)M_PI_2);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);
    __m256 vec_zero = _mm256_set1_ps(0.0f);
    __m256 vec_neg1 = _mm256_set1_ps(-1.0f);

    // Costanti di Maclaurin (da 1/3! a 1/19!)
    __m256 c3 = _mm256_set1_ps(0.16666667f);
    __m256 c5 = _mm256_set1_ps(-8.333333e-3f);
    __m256 c7 = _mm256_set1_ps(1.984127e-4f);
    __m256 c9 = _mm256_set1_ps(-2.755732e-6f);
    __m256 c11 = _mm256_set1_ps(2.505211e-8f);
    __m256 c13 = _mm256_set1_ps(-1.605904e-10f);
    __m256 c15 = _mm256_set1_ps(7.647164e-13f);
    __m256 c17 = _mm256_set1_ps(-2.811457e-15f);
    __m256 c19 = _mm256_set1_ps(8.220635e-18f);

    int i;
    for (i = 0; i <= size - 8; i += 8)
    {
        __m256 y = _mm256_load_ps(&in[i]);

        // x = y - PI/2
        __m256 x = _mm256_sub_ps(y, vec_pi_2);

        // mask = x > PI/2
        __m256 mask = _mm256_cmp_ps(x, vec_pi_2, _CMP_GT_OS);

        // x_sub = x - PI
        __m256 x_sub = _mm256_sub_ps(x, vec_pi);

        // Applica condizionatamente la sottrazione
        x = _mm256_blendv_ps(x, x_sub, mask);

        // Inizializza cos = -x
        __m256 cos_val = _mm256_sub_ps(vec_zero, x);

        // Calcolo potenze senza Horner
        __m256 x2 = _mm256_mul_ps(x, x);

        __m256 x3 = _mm256_mul_ps(x2, x);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x3, c3));

        __m256 x5 = _mm256_mul_ps(x3, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x5, c5));

        __m256 x7 = _mm256_mul_ps(x5, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x7, c7));

        __m256 x9 = _mm256_mul_ps(x7, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x9, c9));

        __m256 x11 = _mm256_mul_ps(x9, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x11, c11));

        __m256 x13 = _mm256_mul_ps(x11, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x13, c13));

        __m256 x15 = _mm256_mul_ps(x13, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x15, c15));

        __m256 x17 = _mm256_mul_ps(x15, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x17, c17));

        __m256 x19 = _mm256_mul_ps(x17, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x19, c19));

        // Correzione del segno finale
        __m256 cos_neg = _mm256_mul_ps(cos_val, vec_neg1);
        cos_val = _mm256_blendv_ps(cos_val, cos_neg, mask);

        _mm256_store_ps(&out[i], cos_val);
    }

    // Coda residua
    for (; i < size; i++)
    {
        out[i] = my_cos((double)in[i]);
    }
}

#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <math.h>



double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void bench_custom_avx_pd(size_t n_ops)
{
    double elapsed_secs;
    double perf;
    const size_t ARR_SIZE = 4096;

    // Allocazione allineata per evitare il carico unaligned (loadu) usato nel codice Rust
    double *bench_arr = (double *)_mm_malloc(ARR_SIZE * sizeof(double), 32);

    for (size_t i = 0; i < ARR_SIZE; i++)
    {
        // ATTENZIONE: per i=0, questo genera 'inf'
        bench_arr[i] = 3.67343456363439 / (double)i;
    }

    size_t ops = (size_t)round((double)n_ops / (double)ARR_SIZE);

    // --- PRIMO BENCHMARK: STIMA FREQUENZA TRAMITE ADDIZIONE ---
    __m256d result = _mm256_set1_pd(0.0);
    double start_time = get_time_sec();

    for (size_t i = 0; i < ops; i++)
    {
        // step_by(8) in Rust = salti di 8 double (due registri YMM da 4 double)
        for (size_t j = 0; j < ARR_SIZE; j += 8)
        {
            __m256d val_0 = _mm256_load_pd(&bench_arr[j]);
            __m256d val_1 = _mm256_load_pd(&bench_arr[j + 4]);

            __m256d temp = _mm256_add_pd(val_0, val_1);
            result = _mm256_add_pd(result, temp);
        }
    }

    elapsed_secs = get_time_sec() - start_time;
    perf = (double)n_ops / elapsed_secs;

    // Formula empirica altamente dipendente dalla microarchitettura
    double estimated_frequency = perf / 4.0 * 1.5;

    // Workaround C per stampare e prevenire il Dead Code Elimination (transmute in Rust)
    double res_array[4] __attribute__((aligned(32)));
    _mm256_store_pd(res_array, result);
    printf("%f\n", res_array[0]);
    printf("Estimated frequency: %.2f GHz\n", estimated_frequency * 1e-9);

    // --- SECONDO BENCHMARK: COSENO ---
    result = _mm256_set1_pd(0.0);
    start_time = get_time_sec();

    for (size_t i = 0; i < ops; i++)
    {
        for (size_t j = 0; j < ARR_SIZE; j += 8)
        {
            __m256d val_0 = _mm256_load_pd(&bench_arr[j]);
            __m256d val_1 = _mm256_load_pd(&bench_arr[j + 4]);

            __m256d temp_0 = _mm256_cos_pd(val_0);
            __m256d temp_1 = _mm256_cos_pd(val_1);

            __m256d sum = _mm256_add_pd(temp_0, temp_1);
            result = _mm256_add_pd(result, sum);
        }
    }

    elapsed_secs = get_time_sec() - start_time;

    // 2. Calcola il throughput (operazioni al secondo)
    perf = (double)n_ops / elapsed_secs;

    // 3. Barriera Anti-Ottimizzazione (Obbligatoria)
    // Se non stampi o non usi 'result', -O3 o -Ofast capiranno che il calcolo
    // non serve a nulla e taglieranno via l'intero ciclo for, azzerando i tempi.
    _mm256_store_pd(res_array, result);
    printf("Risultato anti-DCE (ignora): %f\n", res_array[0]);

    // 4. Stampa le vere metriche
    printf("\n=== RISULTATI BENCHMARK COSENO ===\n");
    printf("Tempo totale : %f secondi\n", elapsed_secs);
    printf("Throughput   : %e ops/sec\n", perf);
    // Tempo medio per singola iterazione di coseno (nanosecondi)
    printf("Latenza media: %.2f ns/op\n", (elapsed_secs / (double)n_ops) * 1e9);

    // 5. Libera la memoria
    _mm_free(bench_arr);
}

__m256d _mm256_cos_pd(__m256d y)
{
    __m256d x = _mm256_sub_pd(y, _mm256_set1_pd(M_PI_2));

    __m256d mask = _mm256_cmp_pd(x, _mm256_set1_pd(M_PI_2), _CMP_GT_OS);
    __m256d x_sub = _mm256_sub_pd(x, _mm256_set1_pd(M_PI));
    x = _mm256_blendv_pd(x, x_sub, mask);

    __m256d cos = _mm256_set1_pd(0.0);
    cos = _mm256_sub_pd(cos, x);

    __m256d x2 = _mm256_mul_pd(x, x);

    __m256d x3 = _mm256_mul_pd(x2, x);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x3, _mm256_set1_pd(0.16666666666666666)));

    __m256d x5 = _mm256_mul_pd(x3, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x5, _mm256_set1_pd(-8.333333333333333e-3)));

    __m256d x7 = _mm256_mul_pd(x5, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x7, _mm256_set1_pd(1.98412698412698413e-4)));

    __m256d x9 = _mm256_mul_pd(x7, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x9, _mm256_set1_pd(-2.75573192239858907e-6)));

    __m256d x11 = _mm256_mul_pd(x9, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x11, _mm256_set1_pd(2.50521083854417188e-8)));

    __m256d x13 = _mm256_mul_pd(x11, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x13, _mm256_set1_pd(-1.60590438368216146e-10)));

    __m256d x15 = _mm256_mul_pd(x13, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x15, _mm256_set1_pd(7.64716373181981648e-13)));

    __m256d x17 = _mm256_mul_pd(x15, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x17, _mm256_set1_pd(-2.81145725434552076e-15)));

    __m256d x19 = _mm256_mul_pd(x17, x2);
    cos = _mm256_add_pd(cos, _mm256_mul_pd(x19, _mm256_set1_pd(8.22063524662432972e-18)));

    __m256d cos_neg = _mm256_mul_pd(cos, _mm256_set1_pd(-1.0));
    cos = _mm256_blendv_pd(cos, cos_neg, mask);

    return cos;
}

// L'uso di static inline obbliga GCC a fondere questa funzione nel ciclo chiamante
static inline __m256 _mm256_cos_minimax_ps(__m256 vec_x)
{
    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);
    __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    __m256 x_inv_pi = _mm256_mul_ps(vec_x, vec_inv_pi);
    __m256 v_k = _mm256_round_ps(x_inv_pi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 k_pi = _mm256_mul_ps(v_k, vec_pi);
    vec_x = _mm256_sub_ps(vec_x, k_pi);

    __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);
    __m256 vec_result = _mm256_fmadd_ps(vec_x2, vec_c4, vec_c2);
    vec_result = _mm256_fmadd_ps(vec_result, vec_x2, vec_c0);

    __m256i k_int = _mm256_cvtps_epi32(v_k);
    __m256i sign_mask = _mm256_slli_epi32(k_int, 31);

    return _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask));
}

void bench_my_avx_minimax_ps(size_t n_ops)
{
    double elapsed_secs;
    double perf;
    const size_t ARR_SIZE = 4096; // Sta comodamente in L1 Cache

    // Allocazione a 32 byte per array di float
    float *bench_arr = (float *)_mm_malloc(ARR_SIZE * sizeof(float), 32);

    // Evito la divisione per zero e riempio l'array con valori sensati
    for (size_t i = 0; i < ARR_SIZE; i++)
    {
        bench_arr[i] = (float)i * 0.01f;
    }

    size_t ops = (size_t)round((double)n_ops / (double)ARR_SIZE);

    __m256 result = _mm256_set1_ps(0.0f);
    double start_time = get_time_sec(); // Assicurati di avere la tua get_time() qui

    for (size_t i = 0; i < ops; i++)
    {
        // Passo a 16: carichiamo 16 float (2 registri YMM) per iterazione
        for (size_t j = 0; j < ARR_SIZE; j += 16)
        {
            __m256 val_0 = _mm256_load_ps(&bench_arr[j]);
            __m256 val_1 = _mm256_load_ps(&bench_arr[j + 8]);

            // Chiamata alla TUA funzione appena riadattata
            __m256 temp_0 = _mm256_cos_minimax_ps(val_0);
            __m256 temp_1 = _mm256_cos_minimax_ps(val_1);

            // Accumulo per non far cancellare il calcolo dal compilatore
            __m256 sum = _mm256_add_ps(temp_0, temp_1);
            result = _mm256_add_ps(result, sum);
        }
    }

    elapsed_secs = get_time_sec() - start_time;
    perf = (double)n_ops / elapsed_secs;

    // Barriera Anti-DCE (Dead Code Elimination)
    float res_array[8] __attribute__((aligned(32)));
    _mm256_store_ps(res_array, result);
    printf("Risultato anti-DCE (ignora): %f\n", res_array[0]);

    printf("\n=== RISULTATI BENCHMARK MY_MINIMAX (FLOAT) ===\n");
    printf("Tempo totale : %f secondi\n", elapsed_secs);
    printf("Throughput   : %e ops/sec\n", perf);
    printf("Latenza media: %.2f ns/op\n", (elapsed_secs / (double)n_ops) * 1e9);

    _mm_free(bench_arr);
}

void my_cos_avx_minimax_zero2pi(const float *in, float *out, int size)
{
    // Costanti per le soglie di riduzione
    __m256 vec_pi_2 = _mm256_set1_ps(1.570796327f);  // PI / 2
    __m256 vec_3pi_2 = _mm256_set1_ps(4.712388980f); // 3PI / 2
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);
    __m256 vec_2pi = _mm256_set1_ps((float)M_PI * 2.0f);

    // Il bit di segno dei float (-0.0f)
    __m256 vec_neg_zero = _mm256_set1_ps(-0.0f);

    __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    int i;

    for (i = 0; i <= size - 8; i += 8)
    {
        __m256 vec_x = _mm256_load_ps(&in[i]);

        // --- RIDUZIONE A MASCHERE (Nessun arrotondamento) ---
        // Controlla se siamo nel 2°/3° quadrante ( > PI/2 )
        __m256 mask_1 = _mm256_cmp_ps(vec_x, vec_pi_2, _CMP_GT_OQ);

        // Controlla se siamo nel 4° quadrante ( > 3PI/2 )
        __m256 mask_2 = _mm256_cmp_ps(vec_x, vec_3pi_2, _CMP_GT_OQ);

        // Se mask_1 è vera, prepariamo PI da sottrarre.
        __m256 sub_val = _mm256_blendv_ps(_mm256_setzero_ps(), vec_pi, mask_1);

        // Se mask_2 è vera, sovrascriviamo con 2PI.
        sub_val = _mm256_blendv_ps(sub_val, vec_2pi, mask_2);

        // Traslazione vettoriale pura (niente moltiplicazioni, solo una sottrazione)
        vec_x = _mm256_sub_ps(vec_x, sub_val);

        // --- POLINOMIO MINIMAX ---
        __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);
        __m256 vec_result = _mm256_fmadd_ps(vec_x2, vec_c4, vec_c2);
        vec_result = _mm256_fmadd_ps(vec_result, vec_x2, vec_c0);

        // --- INVERSIONE DI SEGNO LOGICA ---
        // Il coseno è negativo solo se l'input era compreso tra PI/2 e 3PI/2.
        // Meccanicamente: mask_1 è vera MA mask_2 è falsa.
        // L'operatore XOR isola esattamente questo stato.
        __m256 sign_flip_mask = _mm256_xor_ps(mask_1, mask_2);

        // Applica l'XOR direttamente col bit di segno per chiudere il calcolo
        __m256 sign_bit = _mm256_and_ps(sign_flip_mask, vec_neg_zero);
        vec_result = _mm256_xor_ps(vec_result, sign_bit);

        _mm256_store_ps(&out[i], vec_result);
    }

    for (; i < size; i++)
    {
        out[i] = my_cos(in[i]);
    }
}

void rust_cos_avx_port_optimized(const float *in, float *out, int size)
{
    __m256 vec_pi_2 = _mm256_set1_ps((float)M_PI_2);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);
    __m256 vec_zero = _mm256_set1_ps(0.0f);
    __m256 vec_neg1 = _mm256_set1_ps(-1.0f);

    // Costanti di Maclaurin fermate al grado 11
    // Tutto ciò che è inferiore a 10^-7 è stato rimosso.
    __m256 c3 = _mm256_set1_ps(0.16666667f);
    __m256 c5 = _mm256_set1_ps(-8.333333e-3f);
    __m256 c7 = _mm256_set1_ps(1.984127e-4f);
    __m256 c9 = _mm256_set1_ps(-2.755732e-6f);
    __m256 c11 = _mm256_set1_ps(2.505211e-8f);

    int i;
    for (i = 0; i <= size - 8; i += 8)
    {
        __m256 y = _mm256_load_ps(&in[i]);

        // Riduzione rudimentale
        __m256 x = _mm256_sub_ps(y, vec_pi_2);
        __m256 mask = _mm256_cmp_ps(x, vec_pi_2, _CMP_GT_OS);
        __m256 x_sub = _mm256_sub_ps(x, vec_pi);
        x = _mm256_blendv_ps(x, x_sub, mask);

        __m256 cos_val = _mm256_sub_ps(vec_zero, x);

        // Calcolo potenze (Senza Horner)
        __m256 x2 = _mm256_mul_ps(x, x);

        __m256 x3 = _mm256_mul_ps(x2, x);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x3, c3));

        __m256 x5 = _mm256_mul_ps(x3, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x5, c5));

        __m256 x7 = _mm256_mul_ps(x5, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x7, c7));

        __m256 x9 = _mm256_mul_ps(x7, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x9, c9));

        __m256 x11 = _mm256_mul_ps(x9, x2);
        cos_val = _mm256_add_ps(cos_val, _mm256_mul_ps(x11, c11));

        // Correzione del segno
        __m256 cos_neg = _mm256_mul_ps(cos_val, vec_neg1);
        cos_val = _mm256_blendv_ps(cos_val, cos_neg, mask);

        _mm256_store_ps(&out[i], cos_val);
    }

    for (; i < size; i++)
    {
        out[i] = my_cos((double)in[i]);
    }
}