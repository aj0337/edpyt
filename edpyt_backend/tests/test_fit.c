#include "../fit.h"
#include <stdio.h>


void hybrid_true(double complex vals_true[], int nmats, double beta) {
    double complex zi;
    for (int i=0; i<nmats; i++) {
        zi = (2*i+1) * PI/beta * I;
        vals_true[i] = 2*(zi-csqrt(zi*zi-1.));
    }
}


int test_hybrid(void) {
    int nmats = 3000, nbath = 8, iter=0;
    double beta = 70., fret;
    double complex* vals_true = (double complex*)malloc(nmats*sizeof(double complex));
    double* x = (double*)malloc(2*nbath*sizeof(double));
    hybrid_true(vals_true, nmats, beta);
    fit(x, &iter, &fret, nbath, nmats, vals_true, beta);
    printf("Fit completed with %d iterations.\n", iter);
    printf("Current function value is %.6f.\n", fret);
    for (int j=0; j<2*nbath; j++) {
        printf("%.6f ", x[j]);
    }
    printf("\n");
}


int main(int argc, char* argv[]) {
    test_hybrid();
}