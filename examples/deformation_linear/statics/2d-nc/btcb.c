#define WBMAT_NNODES 4

void
btcb(double K[WBMAT_NNODES*2][WBMAT_NNODES*2],
            double b_l[3][WBMAT_NNODES*2], double C[3][3])
{
  int i, j, k, m;
  const int kdim = WBMAT_NNODES*2;
  double c, blkic;
  for (k=0; k<3; k++) {
    for (m=0; m<3; m++) {
      c = C[k][m];
      if (c != 0) {
        c *= fact;
        for (i=0;  i<kdim; i++) {
          blkic = c * b_l[k][i];
          for (j=0;  j<kdim; j++) {
            K[i][j] += blkic * b_l[m][j];
          }
        }
      }
    }
  }
}

int main()
{
double K[WBMAT_NNODES*2][WBMAT_NNODES*2], b_l[3][WBMAT_NNODES*2], C[3][3];
for (i=0;  i<kdim; i++) {
          for (j=0;  j<kdim; j++) {
            K[i][j] =0.0;
          }
        }
for (i=0;  i<kdim; i++) {

}
}
