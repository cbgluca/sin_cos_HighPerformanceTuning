#include "lib.h"

#define N 10000000 // 10 milioni di elementi per avere un tempo misurabile

double my_sin(double x){
    //bringing it ot the right interval [-pi, +pi]
    double test, x_2,x_3,x_5,x_7,x_9;
    int f_3,f_5,f_7,f_9;

    const double inv_pi = 1.0 / M_PI;

    int k = round(x * inv_pi);

    x = x - (k * M_PI);

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

    //x = x - x_3 / f_3 + x_5 / f_5 - x_7 / f_7 + x_9 / f_9;
    return (k & 1) ? -test : test; // if odd sin(x+pi) = -sin(x)
}


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