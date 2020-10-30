#ifndef FREEFUNCTIONS_H_
#define FREEFUNCTIONS_H_

/******************** Free functions definitions ********************/

template<size_t x>
KOKKOS_INLINE_FUNCTION
double norm(RealArray1D<x> a)
{
	return std::sqrt(dot(a, a));
}

template<size_t x>
KOKKOS_INLINE_FUNCTION
double dot(RealArray1D<x> a, RealArray1D<x> b)
{
	double result(0.0);
	for (size_t i=0; i<x; i++)
	{
		result = result + a[i] * b[i];
	}
	return result;
}

KOKKOS_INLINE_FUNCTION
double computeLpcPlus(RealArray1D<2> xp, RealArray1D<2> xpPlus)
{
	RealArray1D<2> npc_plus;
	double lpc_plus;
	npc_plus[0] = 0.5 * (xpPlus[1] - xp[1]);
	npc_plus[1] = 0.5 * (xp[0] - xpPlus[0]);
	lpc_plus = norm(npc_plus);
	return lpc_plus;
}

KOKKOS_INLINE_FUNCTION
RealArray1D<2> computeNpcPlus(RealArray1D<2> xp, RealArray1D<2> xpPlus)
{
	RealArray1D<2> npc_plus;
	double lpc_plus;
	npc_plus[0] = 0.5 * (xpPlus[1] - xp[1]);
	npc_plus[1] = 0.5 * (xp[0] - xpPlus[0]);
	lpc_plus = norm(npc_plus);
	npc_plus = npc_plus / lpc_plus;
	return npc_plus;
}

KOKKOS_INLINE_FUNCTION
double crossProduct2d(RealArray1D<2> a, RealArray1D<2> b)
{
	return a[0] * b[1] - a[1] * b[0];
}

KOKKOS_INLINE_FUNCTION
RealArray1D<2> computeLpcNpc(RealArray1D<2> xp, RealArray1D<2> xpPlus, RealArray1D<2> xpMinus)
{
	return computeLpcPlus(xp, xpPlus) * computeNpcPlus(xp, xpPlus) + computeLpcPlus(xpMinus, xp) * computeNpcPlus(xpMinus, xp);
}

template<size_t N>
KOKKOS_INLINE_FUNCTION
RealArray1D<N> symmetricVector(RealArray1D<N> v, RealArray1D<N> sigma)
{
	return 2.0 * dot(v, sigma) * sigma - v;
}

template<size_t x>
KOKKOS_INLINE_FUNCTION
RealArray1D<x> sumR1(RealArray1D<x> a, RealArray1D<x> b)
{
	return a + b;
}

KOKKOS_INLINE_FUNCTION
double sumR0(double a, double b)
{
	return a + b;
}

KOKKOS_INLINE_FUNCTION
double minR0(double a, double b)
{
	return std::min(a, b);
}

KOKKOS_INLINE_FUNCTION
double divideNoExcept(double a, double b) {
  if (std::fabs(b) < 1.0E-12)
    return 0.0;
  else
    return a / b;
}

#endif /* FREEFUNCTIONS_H_ */
