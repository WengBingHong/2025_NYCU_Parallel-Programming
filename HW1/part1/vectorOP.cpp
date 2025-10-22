#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  const float CLAMP = 9.999999f;

  __pp_vec_float onef = _pp_vset_float(1.0f);
  __pp_vec_float clampf = _pp_vset_float(CLAMP);
  __pp_vec_int zeroi = _pp_vset_int(0);
  __pp_vec_int onei = _pp_vset_int(1);

  __pp_mask maskActive, maskYeq0, maskYneq0, maskCntPos, maskClamp;
  __pp_vec_float x, result, tmp;
  __pp_vec_int y, count, yMinus1, cntNext;

  // For N % VECTOR_WIDTH != 0
  for (int i = N; i < N + VECTOR_WIDTH; i++)
  {
    values[i] = 0.0f;
    exponents[i] = 1;
  }

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // int lanes = (i + VECTOR_WIDTH <= N) ? VECTOR_WIDTH : (N - i);
    // maskActive = _pp_init_ones(lanes);
    maskActive = _pp_init_ones();

    _pp_vload_float(x, values + i, maskActive);
    _pp_vload_int(y, exponents + i, maskActive);

    _pp_veq_int(maskYeq0, y, zeroi, maskActive); // y == 0
    maskYneq0 = _pp_mask_not(maskYeq0); // y != 0

    // 對 y>0 的 lane：result = x
    result = _pp_vset_float(0.0f);
    _pp_vmove_float(result, x, maskYneq0);

    // count = y - 1（僅 y>0 的 lane）
    count = _pp_vset_int(0);
    _pp_vsub_int(yMinus1, y, onei, maskYneq0);
    _pp_vmove_int(count, yMinus1, maskYneq0);

    // 只在 count>0 的 lane 進行乘法累乘與遞減
    while (true)
    {
      _pp_vgt_int(maskCntPos, count, zeroi, maskActive);
      if (_pp_cntbits(maskCntPos) == 0)
        break;

      _pp_vmult_float(tmp, result, x, maskCntPos);
      _pp_vmove_float(result, tmp, maskCntPos);

      _pp_vsub_int(cntNext, count, onei, maskCntPos);
      _pp_vmove_int(count, cntNext, maskCntPos);
    }

    // CLAMP
    _pp_vgt_float(maskClamp, result, clampf, maskActive);
    _pp_vmove_float(result, clampf, maskClamp);

    // y==0 的 lane 覆寫為 1.0
    _pp_vmove_float(result, onef, maskYeq0);

    // write back
    _pp_vstore_float(output + i, result, maskActive);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float vsum = _pp_vset_float(0.0f);

  // 加入 vsum
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_vec_float v;
    __pp_mask maskAll = _pp_init_ones(VECTOR_WIDTH);
    _pp_vload_float(v, values + i, maskAll);

    __pp_vec_float tmp;
    _pp_vadd_float(tmp, vsum, v, maskAll);
    _pp_vmove_float(vsum, tmp, maskAll);
  }

  // 水平加總 log2(VECTOR_WIDTH) 次
  for (int step = VECTOR_WIDTH; step > 1; step >>= 1)
  {
    _pp_hadd_float(vsum, vsum);
    _pp_interleave_float(vsum, vsum);
  }

  // 取出第一個元素即為總和
  float tmp[VECTOR_WIDTH];
  __pp_mask maskAll = _pp_init_ones(VECTOR_WIDTH);
  _pp_vstore_float(tmp, vsum, maskAll);
  return tmp[0];
}