#ifndef COS_H
#define COS_H

#include "lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scalar cosine approximation using range reduction and 4th-degree Taylor polynomial.
 * @param x Input angle in radians.
 * @return Approximate cosine value.
 */
double my_cos(double x);

/**
 * @brief AVX2 vectorized cosine using range reduction and 4th-degree Taylor polynomial.
 *        Processes 8 single-precision floats per vector step.
 * @param in Pointer to aligned input float array.
 * @param out Pointer to aligned output float array.
 * @param size Number of elements to compute.
 */
void my_cos_avx(const float *in, float *out, int size);

/**
 * @brief AVX2 vectorized cosine using range reduction and 4th-degree Minimax polynomial
 *        evaluated via Horner's scheme.
 *        Processes 8 single-precision floats per vector step.
 * @param in Pointer to aligned input float array.
 * @param out Pointer to aligned output float array.
 * @param size Number of elements to compute.
 */
void my_cos_avx_minimax(const float *in, float *out, int size);

/**
 * @brief AVX2 vectorized cosine with 4th-degree Minimax polynomial and 4x loop unrolling.
 *        Processes 32 single-precision floats per loop iteration to saturate execution units.
 * @param in Pointer to aligned input float array.
 * @param out Pointer to aligned output float array.
 * @param size Number of elements to compute.
 */
void my_cos_avx_minimax_unrolled(const float *in, float *out, int size);

#ifdef __cplusplus
}
#endif

#endif // COS_H
