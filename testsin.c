#include "lib.h"

double my_sin(double);
void my_sin_avx(const float*, float*, int);

double get_time() {
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


    int N = 10000000; // 10 milioni di test
    printf("Generazione di %d valori casuali in corso...\n", N);

    // Alloco memoria per gli array
    float *in = _mm_malloc(N * sizeof(float), 32);
    float *out_std = _mm_malloc(N * sizeof(float), 32);
    float *out_my = _mm_malloc(N * sizeof(float), 32);

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
        in[i] = ((double)rand() / RAND_MAX) * 20.0 * M_PI - 10.0 * M_PI;
    }

    // --- TEST LIBRERIA STANDARD ---
    double start_std = get_time();
    for (int i = 0; i < N; i++) {
        out_std[i] = sin(in[i]); // Nota: sin() per double, sinf() per float
    }
    double end_std = get_time();
    double time_std = end_std - start_std;

    // --- TEST TUA FUNZIONE ---
    double start_my = get_time();
    // for (int i = 0; i < N; i++) {
    //     out_my[i] = my_sin(in[i]);
    // }

    my_sin_avx(in, out_my, N);

    double end_my = get_time();
    double time_my = end_my - start_my;

    // --- CALCOLO ERRORE MASSIMO ---
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(out_std[i] - out_my[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    // --- STAMPA RISULTATI ---
    printf("\n=== RISULTATI BENCHMARK ===\n");
    printf("Tempo Standard (math.h): %f secondi\n", time_std);
    printf("Tempo my_sin           : %f secondi\n", time_my);
    
    if (time_my < time_std) {
        printf("Performance            : La tua è più VELOCE di %.2fx!\n", time_std / time_my);
    } else {
        printf("Performance            : La tua è più LENTA di %.2fx\n", time_my / time_std);
    }
    
    printf("Errore Massimo         : %e\n", max_error);

    _mm_free(in); _mm_free(out_std); _mm_free(out_my);
    return 0;
}



double my_sin(double x){
    //bringing it ot the right interval [-pi, +pi]
    double test, x_2,x_3,x_5,x_7,x_9;
    int f_3,f_5,f_7,f_9;

    const double inv_pi = 1.0 / M_PI;

    int k = round(x * inv_pi);

    x = x - (k * M_PI);


    // while(x > M_PI){
    //     x -= pi_2;
    // }
    // while(x < -M_PI){
    //     x += pi_2;
    // }

    //printf("\ntest x:%lf\n", x);

    x_2= x*x;
    x_3= x_2*x;
    x_5= x_3*x_2;
    //x_7= x_5*x_2;
    //x_9= x_7*x_2;

    f_3 = 6;
    f_5 = 120;
    //f_7 = 5040;
    //f_9 = 362880;

    //test = x - x_3 * (1.0/f_3) + x_5 * (1.0/f_5) - x_7 * (1.0/f_7) + x_9 * (1.0/f_9);
    test = x - x_3 * (1.0/f_3) + x_5 * (1.0/f_5);

    //printf("con 1/f: %lf\n", test);
    //x = x - x_3 / f_3 + x_5 / f_5 - x_7 / f_7 + x_9 / f_9;
    return (k & 1) ? -test : test; // if odd sin(x+pi) = -sin(x)
}


// double power (double num, int exp){
//     for(int i = 1; i<exp; i++){
//         num *= num;
//     }
//     return num;
// }

void my_sin_avx(const float* in, float* out, int size){



    __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    __m256 vec_pi     = _mm256_set1_ps((float)M_PI);

    float c_3 = 1.0f / 6.0f;
    float c_5 = 1.0f / 120.0f;
    __m256 vec_c3 = _mm256_set1_ps(c_3);
    __m256 vec_c5 = _mm256_set1_ps(c_5);

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
        __m256 vec_x3 = _mm256_mul_ps(vec_x2, vec_x);
        __m256 vec_x5 = _mm256_mul_ps(vec_x3, vec_x2);

        __m256 vec_result = _mm256_fnmadd_ps(vec_x3, vec_c3, vec_x);
        vec_result = _mm256_fmadd_ps(vec_x5, vec_c5, vec_result);
        //__m256 vec_term3 = _mm256_mul_ps(vec_x3, vec_c3);
        //__m256 vec_term5 = _mm256_mul_ps(vec_x5, vec_c5);

        //__m256 vec_res_parziale = _mm256_sub_ps(vec_x, vec_term3);
        //__m256 vec_result       = _mm256_add_ps(vec_res_parziale, vec_term5);

        __m256i k_int = _mm256_cvtps_epi32(v_k);
        __m256i sign_mask = _mm256_slli_epi32(k_int, 31);
        vec_result = _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask));

        _mm256_store_ps(&out[i], vec_result);
    }
    //for all the remaining floats that don't fit in a YMM, i use the my_sin() one by one
    for (; i < size; i++) {
        out[i] = my_sin(in[i]); 
    }
    
}