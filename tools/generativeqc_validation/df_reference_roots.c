/* Diagnostic interposer only: retain libcint recurrences, independently
 * validated quadruple-precision moment solver replaces fitted Rys roots. */
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
extern int CINTqrys_jacobi(int, double, double, double*, double*);
extern int CINTqrys_laguerre(int, double, double, double*, double*);
static _Atomic unsigned long calls;
int CINTrys_roots(int n, double x, double* u, double* w) {
  atomic_fetch_add(&calls, 1);
  int status = x <= 20. ? CINTqrys_jacobi(n, x, 0., u, w) : CINTqrys_laguerre(n, x, 0., u, w);
  if (status) {
    fprintf(stderr, "accurate roots failed: %d %.17g\n", n, x);
    abort();
  }
  return 0;
}
unsigned long accurate_root_calls(void) { return atomic_load(&calls); }
