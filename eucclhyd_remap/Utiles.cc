#include <array>  // for array, array<>::value_type

#include "Eucclhyd.h"             // for Eucclhyd
#include "types/MathFunctions.h"  // for det

RealArray2D<2, 2> Eucclhyd::inverse(RealArray2D<2, 2> a) {
  double alpha = 1.0 / MathFunctions::det(a);
  return {
      {a[1][1] * alpha, -a[0][1] * alpha, -a[1][0] * alpha, a[0][0] * alpha}};
}

double Eucclhyd::crossProduct2d(RealArray1D<2> a, RealArray1D<2> b) {
  return a[0] * b[1] - a[1] * b[0];
}
