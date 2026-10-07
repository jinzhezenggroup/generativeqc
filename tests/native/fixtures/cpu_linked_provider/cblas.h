#pragma once
enum {
  CblasRowMajor = 101,
  CblasColMajor = 102,
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasUpper = 121,
  CblasLower = 122,
  CblasNonUnit = 131,
  CblasUnit = 132,
  CblasLeft = 141,
  CblasRight = 142
};
extern "C" {
int openblas_set_num_threads_local(int);
int openblas_get_num_threads();
void openblas_set_num_threads(int);
void cblas_dgemm(int, int, int, int, int, int, double, const double*, int, const double*, int,
                 double, double*, int);
void cblas_dgemv(int, int, int, int, double, const double*, int, const double*, int, double,
                 double*, int);
void cblas_dger(int, int, int, double, const double*, int, const double*, int, double*, int);
void cblas_dsymm(int, int, int, int, int, double, const double*, int, const double*, int, double,
                 double*, int);
void cblas_dsyr(int, int, int, double, const double*, int, double*, int);
void cblas_dsyr2(int, int, int, double, const double*, int, const double*, int, double*, int);
void cblas_dsyrk(int, int, int, int, int, double, const double*, int, double, double*, int);
void cblas_dsyr2k(int, int, int, int, int, double, const double*, int, const double*, int, double,
                  double*, int);
void cblas_dtrsm(int, int, int, int, int, int, int, double, const double*, int, double*, int);
void cblas_dtrmm(int, int, int, int, int, int, int, double, const double*, int, double*, int);
}
