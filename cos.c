#include "lib.h"

#define N 10000000 // 10 milioni di elementi per avere un tempo misurabile

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