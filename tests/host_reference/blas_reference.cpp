#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex>
#include <inttypes.h>

#include <eigen_helper.h>

#include <cassert>

#include "util_quda.h"
#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "blas_lapack.h"

template <typename T> using complex = std::complex<T>;

static void fillEigenArray(MatrixXcd &EigenArr, complex<double> *arr, int rows, int cols, int ld, int offset)
{
  int counter = offset;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      EigenArr(i, j) = arr[counter];
      counter++;
    }
    counter += (ld - cols);
  }
}

void prepare_ref_array(void *array, int batches, uint64_t array_size, size_t data_size, QudaBLASDataType data_type)
{
  memset(array, 0, batches * array_size * data_size);
  // Populate the real part with rands
  for (uint64_t i = 0; i < 2 * array_size * batches; i += 2) { ((double *)array)[i] = rand() / (double)RAND_MAX; }
  // Populate the imaginary part with rands if needed
  if (data_type == QUDA_BLAS_DATATYPE_C || data_type == QUDA_BLAS_DATATYPE_Z) {
    for (uint64_t i = 1; i < 2 * array_size * batches; i += 2) { ((double *)array)[i] = rand() / (double)RAND_MAX; }
  }
}

void copy_array(void *array_out, void *array_in, int batches, uint64_t array_size, size_t data_out_size,
                QudaBLASDataType data_type)
{
  // Copy the real part only
  if (data_type == QUDA_BLAS_DATATYPE_S || data_type == QUDA_BLAS_DATATYPE_D) {
    for (uint64_t i = 0; i < 2 * array_size * batches; i += 2) {
      if (data_out_size == sizeof(float))
        ((float *)array_out)[i / 2] = ((double *)array_in)[i];
      else if (data_out_size == sizeof(double))
        ((double *)array_out)[i / 2] = ((double *)array_in)[i];
      else
        errorQuda("Unsupported data out size %lu", data_out_size);
    }
  }
  // Copy both the real and the imaginary parts
  if (data_type == QUDA_BLAS_DATATYPE_C || data_type == QUDA_BLAS_DATATYPE_Z) {
    for (uint64_t i = 0; i < 2 * array_size * batches; i++) {
      if (data_out_size == sizeof(float))
        ((float *)array_out)[i] = ((double *)array_in)[i];
      else if (data_out_size == sizeof(double))
        ((double *)array_out)[i] = ((double *)array_in)[i];
      else
        errorQuda("Unsupported data out size %lu", data_out_size);
    }
  }
}

// the dumbest possible option that doesn't rely on eigen and cuBLAS interface being right
static double loopVerify( void *A_data, void *B_data, void *C_data_copy,
			  QudaBLASParam blas_param )
{
  // row major batched, strided, non-transposed MMUL for doubles
  double *pA = (double*)A_data , *pB = (double*)B_data , *pC = (double*)C_data_copy ; 
  for( int p = 0 ; p < blas_param.batch_count ; p++ ) {
    for( int m = 0 ; m < blas_param.m ; m++ ) {
      for( int n = 0 ; n < blas_param.n ; n++ ) {
	double Sum = 0. ;
	for( int k = 0 ; k < blas_param.k ; k++ ) {
	  Sum += pA[ k + blas_param.lda*m + p*blas_param.a_stride ]*pB[ n + blas_param.ldb*k + blas_param.b_stride ] ;
	}
	pC[ n + m*blas_param.ldc + p*blas_param.c_stride ] = Sum ;
      }
    }
  }
}

double blasGEMMQudaVerify(void *arrayA, void *arrayB, void *arrayC,
			  uint64_t refA_size, uint64_t refB_size, uint64_t refC_size,
			  QudaBLASParam blas_param)
{
  // data is on the host
  double C_eigen[ refC_size ] , C_loop[ refC_size ] ;

  loopVerify( arrayA , arrayB , C_loop , blas_param ) ;

  quda::blas_lapack::generic::stridedBatchGEMM( arrayA, arrayB , arrayC, blas_param, QUDA_CPU_FIELD_LOCATION ) ;

  // do the copies to the GPU

  // blah blah blah
  
}

double blasLUInvEigenVerify(void *ref_array, void *dev_inv_array, uint64_t array_size, QudaBLASParam *blas_param)
{

  // Sanity checks on parameters
  //-------------------------------------------------------------------------
  // If the batch value is non-positve, we error out
  if (blas_param->batch_count <= 0) { errorQuda("Batches must be positive: batches=%d", blas_param->batch_count); }
  //-------------------------------------------------------------------------

  // Parse parameters for Eigen
  //-------------------------------------------------------------------------
  // Problem parameters
  int offset = 0;
  int batches = blas_param->batch_count;
  int mat_rank = blas_param->inv_mat_size;

  // Eigen objects to store data
  MatrixXcd ref = MatrixXd::Zero(mat_rank, mat_rank);
  MatrixXcd ref_inv = MatrixXd::Zero(mat_rank, mat_rank);
  MatrixXcd dev_inv = MatrixXd::Zero(mat_rank, mat_rank);
  MatrixXcd inv_resid = MatrixXd::Zero(mat_rank, mat_rank);

  // Pointers to data
  complex<double> *ref_ptr = (complex<double> *)(&ref_array)[0];
  complex<double> *dev_inv_ptr = (complex<double> *)(&dev_inv_array)[0];

  printfQuda("Computing Eigen matrix operation ref_inv{%lu,%lu} = ref{%lu,%lu}^(-1)\n", ref.rows(), ref.cols(),
             ref_inv.rows(), ref_inv.cols());

  double max_relative_deviation = 0.0;
  for (int batch = 0; batch < batches; batch++) {

    // Populate Eigen objects
    fillEigenArray(ref, ref_ptr, mat_rank, mat_rank, mat_rank, offset);
    fillEigenArray(dev_inv, dev_inv_ptr, mat_rank, mat_rank, mat_rank, offset);

    // Check Eigen result against blas
    ref_inv = ref.inverse();
    inv_resid = dev_inv - ref_inv;

    double deviation = inv_resid.norm();
    double relative_deviation = deviation / ref_inv.norm();
    max_relative_deviation = std::max(max_relative_deviation, relative_deviation);

    printfQuda("batch %d: (ref_inv - dev_inv) Frobenius norm = %e. Relative deviation = %e\n", batch, deviation,
               relative_deviation);

    offset += array_size;
  }

  return max_relative_deviation;
}

double blasLUInvQudaVerify(void *ref_array, void *dev_array_inv, uint64_t array_size, QudaBLASParam *blas_param)
{
  // Reference data is always in complex double, but real data may
  // be supported in the future.
  size_t data_out_size = sizeof(double);
  int re_im = 2;

  int batches = blas_param->batch_count;

  // Copy data from problem sized array to reference sized array.
  void *dev_array_inv_copy = pinned_malloc(array_size * re_im * data_out_size * batches);

  size_t data_in_size = 0;
  switch (blas_param->data_type) {
  case QUDA_BLAS_DATATYPE_C: data_in_size = sizeof(float); break;
  case QUDA_BLAS_DATATYPE_Z: data_in_size = sizeof(double); break;
  case QUDA_BLAS_DATATYPE_S:
  case QUDA_BLAS_DATATYPE_D:
  default: errorQuda("Unsupported data type %d\n", blas_param->data_type);
  }

  for (uint64_t i = 0; i < 2 * array_size * batches; i++) {
    if (data_in_size == sizeof(float))
      ((double *)dev_array_inv_copy)[i] = ((float *)dev_array_inv)[i];
    else if (data_in_size == sizeof(double))
      ((double *)dev_array_inv_copy)[i] = ((double *)dev_array_inv)[i];
    else
      errorQuda("Unsupported data-in size %lu", data_in_size);
  }

  auto deviation = blasLUInvEigenVerify(ref_array, dev_array_inv_copy, array_size, blas_param);

  host_free(dev_array_inv_copy);
  return deviation;
}
