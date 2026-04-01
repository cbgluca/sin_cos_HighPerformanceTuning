#include "lib.h"

void extract_float(float x, float* m, float* e, uint32_t* boh)
{
    // 1. Inganniamo il C: "Tratta questo float come un intero a 32 bit"
    uint32_t x_int = *(uint32_t *)&x;

    // Spingiamo via i 23 bit della mantissa verso destra e togliamo il bias di 127
    int e_int = (x_int >> 23) - 127;
    *e = (float)e_int; // Ora 'e' è un numero normale, es: 3.0f, -2.0f, ecc.

    // Spegniamo il vecchio esponente (AND logico) e ci incolliamo l'esponente 127 (OR logico)
    // 0x007FFFFF è la maschera per tenere i 23 bit a destra.
    // 0x3F800000 è il numero 127 shiftato nella posizione dell'esponente (rappresenta 1.0f).
    *boh = (x_int & 0x007FFFFF) | 0x3F800000;

    // Riconvertiamo la maschera manipolata in float
    *m = *(float *)boh;
}


void my_log_avx(const float* in, float* out, int size){
    __m256 vec_ln2 = _mm256_set1_ps(0.69314718f);
    __m256i vec_mant = _mm256_set1_epi32(0x007FFFFF);
    __m256i vec_res127 = _mm256_set1_epi32(0x3F800000);
    __m256i vec_zero = _mm256_set1_epi32(0);
    int i;

    for (i=0; i <= size - 8; i += 8) {
        __m256 vec_x = _mm256_load_ps(&in[i]);

        // extracion of mantissa and exponent
        __m256i x_int = _mm256_castps_si256(vec_x);
        __m256i e_int = _mm256_srli_epi32(x_int, 23); 
        e_int = _mm256_sub_epi32(e_int, _mm256_set1_epi32(127)); 
        __m256i m_int = _mm256_and_si256(x_int, vec_mant); // Mantissa
        m_int = _mm256_or_si256(m_int, vec_res127); // normalizing mantissa

        __m256 vec_m = _mm256_castsi256_ps(m_int); // mantissa to float
        __m256 vec_e = _mm256_cvtepi32_ps(e_int); // exponent to float


        __m256 m_minus_1 = _mm256_sub_ps(vec_m, _mm256_set1_ps(1.0f));
        __m256 term1 = m_minus_1;
        __m256 term2 = _mm256_mul_ps(_mm256_mul_ps(m_minus_1, m_minus_1), _mm256_set1_ps(-0.5f));
        __m256 term3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(m_minus_1, m_minus_1), m_minus_1), _mm256_set1_ps(0.33333333f));
        
        __m256 log_m = _mm256_add_ps(term1, _mm256_add_ps(term2, term3));

        // log(x) = log(mantissa) + esponente * ln(2)
        __m256 log_x = _mm256_add_ps(log_m, _mm256_mul_ps(vec_e, vec_ln2));
        _mm256_store_ps(&out[i], log_x);
    }
}

    int main()
    {
        float x = 5.75f; // Esempio di numero float
        float m, e;
        uint32_t boh;
        extract_float(x, &m, &e, &boh);
        printf("num: %f\n", x);
        printf("mantissa: %f\n", m);
        printf("esponente: %f\n", e);
        printf("boh (mantissa manipolata): %u\n", boh);
        return 0;
    }
