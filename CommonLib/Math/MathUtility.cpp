#include "MathUtility.h"
#include "CMatrixInv.h"

#include "linalg.h"


double MathUtility::logSumExp(double* lh, double* alpha, int n) {
	int maxIdx = -1;
	double maxVal = 0;
	for (int i = 0; i < n; i++) {
		if (maxIdx == -1 || lh[i] > maxVal) {
			maxIdx = i;
			maxVal = lh[i];
		}
	}

	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += alpha[i] * exp(lh[i] - maxVal);
	}
	double res = maxVal + log(sum);
	return res;
}


double MathUtility::rcond(double* mat, int n) {
	alglib::real_2d_array M;
	M.setcontent(n, n, mat);
	double rcond = alglib::rmatrixrcond1(M, n);
	return rcond;
}

void MathUtility::inv(double* mat, int n) {
	MatrixInv(mat, n);
	return;
}

double MathUtility::prod(double* x, int n) {
	if (n < 0) {
		printf("error in prod, n should > 0\n");
		exit(-1);
	}
	double r = 1;
	for (int i = 0; i < n; i++)
		r *= x[i];
	return r;
}



double MathUtility::det(double* mat, int n) {
	alglib::real_2d_array M;
	//M.setlength(45, 45)
	M.setcontent(n, n, mat);
	double det = alglib::rmatrixdet(M);
	return det;
}

void MathUtility::smallest_eig_sym(double* mat, int n, double* eval, double* evec) {
	
	alglib::real_2d_array M;
	M.setcontent(n, n, mat);
	alglib::real_1d_array w;
	alglib::real_2d_array z;

	alglib::smatrixevdi(M, n, 1, true, 0, 0, w, z);
	*eval = *(w.getcontent());
	for (int i = 0; i < n; i++) {
		evec[i] = z[i][0];
	}

	return;
}

double MathUtility::vdist(double* a, double* b, int len) {
	double r = 0;
	for (int i = 0; i < len; i++) {
		r += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(r);
}

double MathUtility::logSumExp(double* lh, int n) {
	int maxIdx = -1;
	double maxVal = 0;
	for (int i = 0; i < n; i++) {
		if (maxIdx == -1 || lh[i] > maxVal) {
			maxIdx = i;
			maxVal = lh[i];
		}
	}

	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += 1 * exp(lh[i] - maxVal);
	}
	double res = maxVal + log(sum);
	return res;
}