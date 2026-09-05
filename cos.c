#include "cos.h"

/**
 * @brief Scalar implementation of cosine using range reduction and 4th-degree Taylor expansion.
 *
 * Algorithm explanation:
 * 1. Range reduction:
 *    Any input angle x can be expressed as:
 *       x = k * PI + x_reduced, where k = round(x / PI)
 *    This maps x_reduced strictly into the interval [-PI/2, +PI/2].
 * 2. Taylor approximation:
 *    cos(x_reduced) ≈ 1 - (x_reduced^2 / 2!) + (x_reduced^4 / 4!)
 * 3. Sign reconstruction:
 *    Since cos(x_reduced + k * PI) = (-1)^k * cos(x_reduced):
 *    If k is odd, the result is inverted; otherwise it stays positive.
 *
 * @param x Input angle in radians.
 * @return Approximate cosine value.
 */
double my_cos(double x)
{
    const double inv_pi = 1.0 / M_PI;

    // k represents the number of PI half-periods
    int k = (int)round(x * inv_pi);

    // Reduced angle in [-PI/2, +PI/2]
    x = x - ((double)k * M_PI);

    double x_2 = x * x;
    double x_4 = x_2 * x_2;

    // 4th-degree Taylor series: 1 - x^2 / 2 + x^4 / 24
    double result = 1.0 - (x_2 * (1.0 / 2.0)) + (x_4 * (1.0 / 24.0));

    // If k is odd, flip sign; if k is even, keep positive
    return (k & 1) ? -result : result;
}

/**
 * @brief AVX2 vectorized cosine using 4th-degree Taylor expansion.
 *        Processes 8 single-precision floats simultaneously using 256-bit YMM registers.
 *
 * Vector steps:
 * 1. Load 8 floats into YMM register.
 * 2. Multiply by (1 / PI) and round to nearest integer using _MM_FROUND_TO_NEAREST_INT.
 * 3. Compute reduced angle x = x - (k * PI).
 * 4. Compute x^2 and x^4.
 * 5. Evaluate Taylor polynomial using fused multiply-add (FMA) instructions:
 *      result = 1.0 - x^2 * (1/2) + x^4 * (1/24)
 * 6. Convert k to 32-bit integer, shift left by 31 bits to align with IEEE 754 sign bit,
 *    and XOR with the result (toggles the sign bit when k is odd).
 * 7. Store 8 results to output array.
 * 8. Any remaining elements (size % 8) are computed using the scalar fallback.
 *
 * @param in Pointer to input float array.
 * @param out Pointer to destination float array.
 * @param size Total number of elements.
 */
void my_cos_avx(const float *in, float *out, int size)
{
    const __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    const __m256 vec_pi     = _mm256_set1_ps((float)M_PI);
    const __m256 vec_c2     = _mm256_set1_ps(0.5f);        // 1/2!
    const __m256 vec_c4     = _mm256_set1_ps(1.0f / 24.0f); // 1/4!
    const __m256 vec_1      = _mm256_set1_ps(1.0f);

    int i = 0;

    // Process 8 floats per vector iteration
    for (; i + 8 <= size; i += 8)
    {
        __m256 vec_x = _mm256_load_ps(&in[i]);

        // k = round(x / PI)
        __m256 x_inv_pi = _mm256_mul_ps(vec_x, vec_inv_pi);
        __m256 v_k = _mm256_round_ps(x_inv_pi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // x_reduced = x - (k * PI)
        __m256 k_pi = _mm256_mul_ps(v_k, vec_pi);
        vec_x = _mm256_sub_ps(vec_x, k_pi);

        // Compute powers of x: x^2 and x^4
        __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);
        __m256 vec_x4 = _mm256_mul_ps(vec_x2, vec_x2);

        // Evaluate: 1.0 - 0.5 * x^2 + (1/24) * x^4 using FMA
        // fnmadd: -(vec_x2 * vec_c2) + vec_1
        __m256 vec_result = _mm256_fnmadd_ps(vec_x2, vec_c2, vec_1);
        // fmadd: (vec_x4 * vec_c4) + vec_result
        vec_result = _mm256_fmadd_ps(vec_x4, vec_c4, vec_result);

        // Sign correction: if k is odd, flip sign bit (bit 31)
        __m256i k_int = _mm256_cvtps_epi32(v_k);
        __m256i sign_mask = _mm256_slli_epi32(k_int, 31);
        vec_result = _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask));

        _mm256_store_ps(&out[i], vec_result);
    }

    // Scalar fallback loop for remaining elements
    for (; i < size; i++)
    {
        out[i] = (float)my_cos((double)in[i]);
    }
}

/**
 * @brief AVX2 vectorized cosine using 4th-degree Minimax (Chebyshev) polynomial with Horner's scheme.
 *
 * Why Minimax?
 * - Standard Taylor series is centered at 0, which makes error grow rapidly near |x| ≈ PI/2.
 * - The Minimax polynomial distributes the approximation error uniformly across [-PI/2, +PI/2],
 *   substantially reducing the maximum error (from ~2.0e-2 down to ~5.97e-4).
 *
 * Horner's Method:
 * - Polynomial: P(x) = c0 + c2 * x^2 + c4 * x^4
 * - Horner's form: P(x) = c0 + x^2 * (c2 + x^2 * c4)
 * - Requires only 2 FMA instructions instead of separate powers and additions.
 *
 * @param in Pointer to input float array.
 * @param out Pointer to destination float array.
 * @param size Total number of elements.
 */
void my_cos_avx_minimax(const float *in, float *out, int size)
{
    const __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    const __m256 vec_pi     = _mm256_set1_ps((float)M_PI);

    // Minimax coefficients for degree-4 approximation of cos(x) on [-PI/2, PI/2]
    const __m256 vec_c0 = _mm256_set1_ps(0.9994032f);
    const __m256 vec_c2 = _mm256_set1_ps(-0.4955807f);
    const __m256 vec_c4 = _mm256_set1_ps(0.0367916f);

    int i = 0;

    // Process 8 floats per vector iteration
    for (; i + 8 <= size; i += 8)
    {
        __m256 vec_x = _mm256_load_ps(&in[i]);

        // Range reduction: k = round(x / PI)
        __m256 x_inv_pi = _mm256_mul_ps(vec_x, vec_inv_pi);
        __m256 v_k = _mm256_round_ps(x_inv_pi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // x_reduced = x - (k * PI)
        __m256 k_pi = _mm256_mul_ps(v_k, vec_pi);
        vec_x = _mm256_sub_ps(vec_x, k_pi);

        // x^2
        __m256 vec_x2 = _mm256_mul_ps(vec_x, vec_x);

        // Horner's evaluation: c0 + x^2 * (c2 + x^2 * c4)
        // Step 1: (x^2 * c4) + c2
        __m256 vec_result = _mm256_fmadd_ps(vec_x2, vec_c4, vec_c2);
        // Step 2: (result * x^2) + c0
        vec_result = _mm256_fmadd_ps(vec_result, vec_x2, vec_c0);

        // Sign correction via bitwise XOR:
        // Convert k to integer, shift LSB to bit 31, and XOR with the float result
        __m256i k_int = _mm256_cvtps_epi32(v_k);
        __m256i sign_mask = _mm256_slli_epi32(k_int, 31);
        vec_result = _mm256_xor_ps(vec_result, _mm256_castsi256_ps(sign_mask));

        _mm256_store_ps(&out[i], vec_result);
    }

    // Scalar fallback loop for remaining elements
    for (; i < size; i++)
    {
        out[i] = (float)my_cos((double)in[i]);
    }
}

/**
 * @brief Highly optimized AVX2 cosine using Minimax polynomial with 4x loop unrolling.
 *
 * Microarchitectural Optimizations:
 * 1. 4x Loop Unrolling (32 floats / 128 bytes per iteration):
 *    Processes 4 YMM registers in parallel. This breaks serial dependency chains,
 *    hides FMA and load latencies (typically 4-5 cycles), and maximizes instruction-level
 *    parallelism (ILP) on modern out-of-order execution pipelines.
 * 2. Single-instruction Range Reduction:
 *    Uses _mm256_fnmadd_ps(k, PI, x) which computes -(k * PI) + x in a single cycle,
 *    eliminating an explicit multiplication and subtraction.
 * 3. Interleaved Pipeline Execution:
 *    Interleaving independent operations across registers x0..x3 allows CPU execution ports
 *    to remain 100% saturated without stalling on register dependencies.
 *
 * @param in Pointer to input float array (must be 32-byte aligned for maximum throughput).
 * @param out Pointer to destination float array (must be 32-byte aligned).
 * @param size Total number of elements.
 */
void my_cos_avx_minimax_unrolled(const float *in, float *out, int size)
{
    // Constants kept in registers outside the loop to prevent redundant loads
    const __m256 vec_inv_pi = _mm256_set1_ps(1.0f / (float)M_PI);
    const __m256 vec_pi     = _mm256_set1_ps((float)M_PI);
    const __m256 vec_c0     = _mm256_set1_ps(0.9994032f);
    const __m256 vec_c2     = _mm256_set1_ps(-0.4955807f);
    const __m256 vec_c4     = _mm256_set1_ps(0.0367916f);

    int i = 0;

    // Process 32 floats (4 x 8-float YMM registers) per iteration
    for (; i + 32 <= size; i += 32)
    {
        // 1. Independent loads across 4 vector registers
        __m256 x0 = _mm256_load_ps(&in[i]);
        __m256 x1 = _mm256_load_ps(&in[i + 8]);
        __m256 x2 = _mm256_load_ps(&in[i + 16]);
        __m256 x3 = _mm256_load_ps(&in[i + 24]);

        // 2. Compute k = round(x / PI)
        __m256 k0 = _mm256_round_ps(_mm256_mul_ps(x0, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k1 = _mm256_round_ps(_mm256_mul_ps(x1, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k2 = _mm256_round_ps(_mm256_mul_ps(x2, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 k3 = _mm256_round_ps(_mm256_mul_ps(x3, vec_inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // 3. Fused range reduction: x = -(k * PI) + x
        x0 = _mm256_fnmadd_ps(k0, vec_pi, x0);
        x1 = _mm256_fnmadd_ps(k1, vec_pi, x1);
        x2 = _mm256_fnmadd_ps(k2, vec_pi, x2);
        x3 = _mm256_fnmadd_ps(k3, vec_pi, x3);

        // 4. Compute x^2
        __m256 x0_2 = _mm256_mul_ps(x0, x0);
        __m256 x1_2 = _mm256_mul_ps(x1, x1);
        __m256 x2_2 = _mm256_mul_ps(x2, x2);
        __m256 x3_2 = _mm256_mul_ps(x3, x3);

        // 5. Horner's evaluation step 1: (x^2 * c4) + c2
        __m256 res0 = _mm256_fmadd_ps(x0_2, vec_c4, vec_c2);
        __m256 res1 = _mm256_fmadd_ps(x1_2, vec_c4, vec_c2);
        __m256 res2 = _mm256_fmadd_ps(x2_2, vec_c4, vec_c2);
        __m256 res3 = _mm256_fmadd_ps(x3_2, vec_c4, vec_c2);

        // 6. Horner's evaluation step 2: (res * x^2) + c0
        res0 = _mm256_fmadd_ps(res0, x0_2, vec_c0);
        res1 = _mm256_fmadd_ps(res1, x1_2, vec_c0);
        res2 = _mm256_fmadd_ps(res2, x2_2, vec_c0);
        res3 = _mm256_fmadd_ps(res3, x3_2, vec_c0);

        // 7. Extract sign mask from k
        __m256i sign_mask0 = _mm256_slli_epi32(_mm256_cvtps_epi32(k0), 31);
        __m256i sign_mask1 = _mm256_slli_epi32(_mm256_cvtps_epi32(k1), 31);
        __m256i sign_mask2 = _mm256_slli_epi32(_mm256_cvtps_epi32(k2), 31);
        __m256i sign_mask3 = _mm256_slli_epi32(_mm256_cvtps_epi32(k3), 31);

        // 8. Apply sign flip via bitwise XOR
        res0 = _mm256_xor_ps(res0, _mm256_castsi256_ps(sign_mask0));
        res1 = _mm256_xor_ps(res1, _mm256_castsi256_ps(sign_mask1));
        res2 = _mm256_xor_ps(res2, _mm256_castsi256_ps(sign_mask2));
        res3 = _mm256_xor_ps(res3, _mm256_castsi256_ps(sign_mask3));

        // 9. Store 32 computed floats
        _mm256_store_ps(&out[i],      res0);
        _mm256_store_ps(&out[i + 8],  res1);
        _mm256_store_ps(&out[i + 16], res2);
        _mm256_store_ps(&out[i + 24], res3);
    }

    // Scalar fallback loop for remaining elements
    for (; i < size; i++)
    {
        out[i] = (float)my_cos((double)in[i]);
    }
}