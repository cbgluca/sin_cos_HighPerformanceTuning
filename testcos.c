#include "lib.h"

double my_cos(double);
void my_cos_avx(const float*, float*, int);
void my_cos_avx_minimax(const float*, float *, int);
void my_cos_avx_minimax_unrolled(const float *, float *, int);
void my_cos_avx_minimax_cody(const float *, float *, int);
void avx_cos_minimax_array(const float *, float *, size_t);
void my_cos_avx_speed_only(const float *, float *, int);

    double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(){
    
    // printf("Insert value for sin: ");
    //double x;
    // scanf("%lf", &x);
    // double result = sinf(x);
    // printf("real sin(%lf) = %lf\n", x, result);


    // double result_approx = my_sin(x);
    // printf("approximate sin(%lf) = %lf\n", x, result_approx);

    int N = 100000000; // 10000000    4096
    int iterazioni = 100000;


    printf("Generazione di %d valori casuali in corso...\n", N);

    // Alloco memoria per gli array
    float *in = _mm_malloc(N * sizeof(float), 32);
    float *out_std = _mm_malloc(N * sizeof(float), 32);
    float *out_my = _mm_malloc(N * sizeof(float), 32);
    float *out_my2 = _mm_malloc(N * sizeof(float), 32);

    // // random tra -1mln e 1mln
    // for (int i = 0; i < N; i++) {
    //     // 1. Genera un numero tra 0.0f e 1.0f
    //     float frazione = (float)rand() / (float)RAND_MAX; 
        
    //     // 2. Moltiplicalo per un range gigantesco (es. 2 milioni)
    //     float ampiezza = 2000000.0f;
        
    //     // 3. Spostalo per avere sia positivi che negativi (da -1 milione a +1 milione)
    //     in[i] = (frazione * ampiezza) - (ampiezza / 2.0f);
    // }

    //between +10pi e -10pi
    for (int i = 0; i < N; i++) {
        in[i] = ((double)rand() / RAND_MAX) * 4.0 * M_PI - 2.0 * M_PI;
    }

    // // --- TEST LIBRERIA STANDARD ---
    // double start_std = get_time();

    // for (int k = 0; k < iterazioni; k++)
    // {
    // #pragma omp simd
    //     for (int i = 0; i < N; i++)
    //     {
    //         out_std[i] = cosf(in[i]); // cosf, NON cos
    //     }
    // }
    // double end_std = get_time();

    // double time_std = end_std - start_std;

    //--- OLD STANDARD ---
    double start_std = get_time();

    for (int i = 0; i < N; i++)
    {
        out_std[i] = cosf(in[i]); // cosf, NON cos  
    }
    double end_std = get_time();

    double time_std = end_std - start_std;

    // // --- TEST TUA FUNZIONE ---
    // double start_my = get_time();

    // for (int k = 0; k < iterazioni; k++)
    // {
    //     my_cos_avx_minimax(in, out_my, N);
    // }

    // double end_my = get_time();
    // double time_my = end_my - start_my;

    // //test seconda funz
    // double start_my2 = get_time();

    // for (int k = 0; k < iterazioni; k++)
    // {
    //     my_cos_avx_minimax_unrolled(in, out_my, N);
    // }

    // double end_my2 = get_time();
    // double time_my2 = end_my2 - start_my2;

    // --- OLD TEST ---
    double start_my = get_time();
    
    my_cos_avx_speed_only(in, out_my, N);
    
    double end_my = get_time();
    double time_my = end_my - start_my;



    // test seconda funz
    double start_my2 = get_time();


    avx_cos_minimax_array(in, out_my2, N);
 

    double end_my2 = get_time();
    double time_my2 = end_my2 - start_my2;

    // --- CALCOLO ERRORE MASSIMO ---
    double max_error = 0.0;
    double max_error2 = 0.0;
    for (int i = 0; i < N; i++)
    {
        double error = fabs(out_std[i] - out_my[i]);
        double error2 = fabs(out_std[i] - out_my2[i]);
        if (error > max_error)
        {
            max_error = error;
        }
        if(error2 > max_error2){
            max_error2 = error2;
        }

    }

    // --- STAMPA RISULTATI ---
    printf("\n=== RISULTATI BENCHMARK ===\n");
    printf("Tempo Standard (math.h): %f secondi\n", time_std);
    printf("Tempo my_cos_roll           : %f secondi\n", time_my);
    printf("Tempo my_cos_minmax           : %f secondi\n", time_my2);

    if (time_my < time_std) {
        printf("Performance            : La tua è più VELOCE di %.2fx!\n", time_std / time_my);
    } else {
        printf("Performance            : La tua è più LENTA di %.2fx\n", time_my / time_std);
    }

    if (time_my2 < time_std)
    {
        printf("Performance            : La tua è più VELOCE di %.2fx!\n", time_std / time_my2);
    }
    else
    {
        printf("Performance            : La tua è più LENTA di %.2fx\n", time_my2 / time_std);
    }

    printf("Errore Massimo         : %e   errore 2: %e\n", max_error, max_error2);
    




    _mm_free(in); _mm_free(out_std); _mm_free(out_my);
    return 0;
}



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
    // 1. Costanti fuori dal ciclo per evitare memory load ridondanti
    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi = _mm256_set1_ps((float)M_PI);
    __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    int i;

    // 2. Loop Unrolling x4: elabora 32 float per iterazione
    for (i = 0; i <= size - 32; i += 32)
    {
        // Caricamento indipendente per saturare la banda L1
        __m256 x0 = _mm256_load_ps(&in[i]);
        __m256 x1 = _mm256_load_ps(&in[i + 8]);
        __m256 x2 = _mm256_load_ps(&in[i + 16]);
        __m256 x3 = _mm256_load_ps(&in[i + 24]);

        // Calcolo k = round(x * inv_pi)
        __m256 k0 = _mm256_round_ps(_mm256_mul_ps(x0, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k1 = _mm256_round_ps(_mm256_mul_ps(x1, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k2 = _mm256_round_ps(_mm256_mul_ps(x2, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k3 = _mm256_round_ps(_mm256_mul_ps(x3, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // x = x - k*pi ottimizzato in una singola FMA: x = -(k*pi) + x
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
        // Alternare le istruzioni nasconde la latenza di FMA
        __m256 res0 = _mm256_fmadd_ps(x0_2, vec_c4, vec_c2);
        __m256 res1 = _mm256_fmadd_ps(x1_2, vec_c4, vec_c2);
        __m256 res2 = _mm256_fmadd_ps(x2_2, vec_c4, vec_c2);
        __m256 res3 = _mm256_fmadd_ps(x3_2, vec_c4, vec_c2);

        res0 = _mm256_fmadd_ps(res0, x0_2, vec_c0);
        res1 = _mm256_fmadd_ps(res1, x1_2, vec_c0);
        res2 = _mm256_fmadd_ps(res2, x2_2, vec_c0);
        res3 = _mm256_fmadd_ps(res3, x3_2, vec_c0);

        // Maschera del segno
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

    // Coda per gli elementi rimanenti
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