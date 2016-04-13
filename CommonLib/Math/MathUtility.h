#ifndef WJQ_MATH_UTILITY_H
#define WJQ_MATH_UTILITY_H

#define MAX_EXP_VAL 300
class MathUtility {
public:
	static double rcond(double* mat, int n);

	static void inv(double* mat, int n);

	static double det(double* mat, int n);

	static void smallest_eig_sym(double* mat, int n, double* eval, double* evec);

	static double vdist(double* a, double* b, int len);

	static double prod(double* x, int n);

	static double logSumExp(double* lh, double* alpha, int n);

	static double logSumExp(double* lh, int n);

};

#endif