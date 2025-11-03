/**
 * Exponential Lookup Table for INT8 Softmax (IMPROVED VERSION)
 * Pre-computed values: exp(x) * 1048576 for x in [0, -127]
 * 
 * Usage: EXP_LUT_INT8[-x] where x is in range [-127, 0]
 * Memory: 512 bytes (128 entries * 4 bytes)
 * Scale: Divide by 1048576 to get actual exp(x) value
 */

#ifndef EXP_LUT_INT8_H
#define EXP_LUT_INT8_H

#include <stdint.h>

#define EXP_LUT_SCALE 1048576

static const int32_t EXP_LUT_INT8[128] = {
       1048576,     385749,     141909,      52205,      19205,       7065,       2599,        956,
           351,        129,         47,         17,          6,          2,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1,
             1,          1,          1,          1,          1,          1,          1,          1
};

#endif // EXP_LUT_INT8_H
