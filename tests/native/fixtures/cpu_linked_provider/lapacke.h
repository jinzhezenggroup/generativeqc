#pragma once
enum { LAPACK_ROW_MAJOR = 101 };
extern "C" {
int LAPACKE_dpotrf(int, char, int, double*, int);
int LAPACKE_dsyevd(int, char, char, int, double*, int, double*);
}
