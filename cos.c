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


void my_cos_avx(const float* in, float* out, int size)
{ // very inconsistent from 1.2x slower to 1.1x faster, max err 1.996897e-02

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
{ // consistent 1.1x faster, max err: 5.970597e-04

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

        //x is now in [-pi/2 , +pi/2]
        //chebychev coefficients


        __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);

        // Calcolo di P(x) = c_0 + c_2 * x^2 + c_4 * x^4
        // HORNER METHOD to skip 1 operation: P(x) = c_0 + x^2 * (c_2 + x^2 * c_4)

        __m256 vec_result = _mm256_fmadd_ps(vec_x2, vec_c4, vec_c2);
        vec_result = _mm256_fmadd_ps(vec_result, vec_x2, vec_c0);


        __m256i k_int = _mm256_cvtps_epi32(v_k); //converts from float to epi32
        __m256i sign_mask = _mm256_slli_epi32(k_int, 31); //left shit, to get sign
        vec_result = _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask)); //cats is a bypass to make xor work. tricks xor to think its a float and not a epi32

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