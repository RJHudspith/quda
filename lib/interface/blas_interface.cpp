#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <tune_quda.h>

using namespace quda;

// Forward declarations for profiling and parameter checking
// The helper functions are defined in interface_quda.cpp
TimeProfile &getProfileBLAS();
void checkBLASParam(QudaBLASParam &param);

void blasGEMMQuda( const void *arrayA, const void *arrayB, void *arrayC,
		   const QudaBoolean use_native, QudaBLASParam blas_param)
{
  getProfileBLAS().TPSTART(QUDA_PROFILE_TOTAL);
  checkBLASParam(blas_param);

  // if we want to do this only on the host we call this
  if (use_native == QUDA_BOOLEAN_FALSE) {
    getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
    blas_lapack::generic::stridedBatchGEMM(arrayA, arrayB, arrayC, blas_param, QUDA_CPU_FIELD_LOCATION);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);

  // otherwise we need to allocate on the device and copy over
  } else {
    getProfileBLAS().TPSTART(QUDA_PROFILE_INIT);
    
    size_t data_size = 4 ;
    switch( blas_param.data_type ) {
    case QUDA_BLAS_DATATYPE_S : data_size = 4  ; break ;
    case QUDA_BLAS_DATATYPE_D : data_size = 8  ; break ;
    case QUDA_BLAS_DATATYPE_C : data_size = 8  ; break ;
    case QUDA_BLAS_DATATYPE_Z : data_size = 16 ; break ;
    default :
      errorQuda( "Unknown blas data_type %d" , blas_param.data_type ) ;
      break ;
    }
    // Extract data from the param struct for device malloc looks a bit funny because of
    // the changes caused by the batching
    size_t arrayA_size = 0, arrayB_size = 0, arrayC_size = 0;
    if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
      if (blas_param.trans_a == QUDA_BLAS_OP_N) {
        arrayA_size = blas_param.m + blas_param.lda * (blas_param.k-1) ; 
      } else {
	arrayA_size = blas_param.k + blas_param.lda * (blas_param.m-1) ; 
      }
      if (blas_param.trans_b == QUDA_BLAS_OP_N) {
        arrayB_size = blas_param.k + blas_param.ldb * (blas_param.n-1) ; 
      } else {
        arrayB_size = blas_param.n + blas_param.ldb * (blas_param.k-1) ;
      }
      arrayC_size = blas_param.m + blas_param.ldc * (blas_param.n-1) ; 
    } else {
      if (blas_param.trans_a == QUDA_BLAS_OP_N) {
        arrayA_size = blas_param.k + blas_param.lda * (blas_param.m-1); 
      } else {
        arrayA_size = blas_param.m + blas_param.lda * (blas_param.k-1); 
      }
      if (blas_param.trans_b == QUDA_BLAS_OP_N) {
	arrayB_size = blas_param.n + blas_param.ldb * (blas_param.k-1) ;
      } else {
	arrayB_size = blas_param.k + blas_param.ldb * (blas_param.m-1) ;
      }
      arrayC_size = blas_param.n + blas_param.ldc * (blas_param.m-1) ;
    }
    arrayA_size += (blas_param.batch_count-1)*blas_param.a_stride ;
    arrayB_size += (blas_param.batch_count-1)*blas_param.b_stride ;
    arrayC_size += (blas_param.batch_count-1)*blas_param.c_stride ;
    
    const size_t A_bytes = arrayA_size * data_size;
    const size_t B_bytes = arrayB_size * data_size;
    const size_t C_bytes = arrayC_size * data_size;
    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("A_Gbtyes = %f, B_Gbtyes = %f, C_Gbtyes = %f\n", 1.0 * A_bytes / std::pow(1024, 3),
                 1.0 * B_bytes / std::pow(1024, 3), 1.0 * C_bytes / std::pow(1024, 3));
    }
    void *A_d = pool_device_malloc(A_bytes);
    void *B_d = pool_device_malloc(B_bytes);
    void *C_d = pool_device_malloc(C_bytes);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("QUDA: arrays allocated successfully.\n");
    getProfileBLAS().TPSTOP(QUDA_PROFILE_INIT);

    // Transfer host data to device
    getProfileBLAS().TPSTART(QUDA_PROFILE_H2D);
    qudaMemcpy(A_d, arrayA, A_bytes, qudaMemcpyHostToDevice);
    qudaMemcpy(B_d, arrayB, B_bytes, qudaMemcpyHostToDevice);
    qudaMemcpy(C_d, arrayC, C_bytes, qudaMemcpyHostToDevice);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("QUDA: arrays copied successfully.\n");
    getProfileBLAS().TPSTOP(QUDA_PROFILE_H2D);

    // Compute Batched GEMM
    getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);

    blas_lapack::native::stridedBatchGEMM(A_d, B_d, C_d, blas_param, QUDA_CUDA_FIELD_LOCATION);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("BatchGEMM success!\n");
    getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);

    // Copy device C array back to host
    getProfileBLAS().TPSTART(QUDA_PROFILE_D2H);
    qudaMemcpy(arrayC, C_d, C_bytes, qudaMemcpyDeviceToHost);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_D2H);

    // Clean up
    getProfileBLAS().TPSTART(QUDA_PROFILE_FREE);
    pool_device_free(A_d);
    pool_device_free(B_d);
    pool_device_free(C_d);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_FREE);
  }

  getProfileBLAS().TPSTOP(QUDA_PROFILE_TOTAL);
  saveTuneCache();
}

void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *blas_param)
{
  getProfileBLAS().TPSTART(QUDA_PROFILE_TOTAL);
  checkBLASParam(*blas_param);

  getProfileBLAS().TPSTART(QUDA_PROFILE_INIT);
  const int n = blas_param->inv_mat_size;
  const uint64_t batches = blas_param->batch_count;
  QudaPrecision prec = QUDA_INVALID_PRECISION;
  switch (blas_param->data_type) {
  case QUDA_BLAS_DATATYPE_Z: prec = QUDA_DOUBLE_PRECISION; break;
  case QUDA_BLAS_DATATYPE_C: prec = QUDA_SINGLE_PRECISION; break;
  case QUDA_BLAS_DATATYPE_D:
  case QUDA_BLAS_DATATYPE_S:
  default: errorQuda("LU inversion not supported for data type %d", blas_param->data_type);
  }
  getProfileBLAS().TPSTOP(QUDA_PROFILE_INIT);

  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
  if (use_native == QUDA_BOOLEAN_FALSE)
    blas_lapack::generic::BatchInvertMatrix(Ainv, A, n, batches, prec, QUDA_CPU_FIELD_LOCATION);
  else
    blas_lapack::native::BatchInvertMatrix(Ainv, A, n, batches, prec, QUDA_CPU_FIELD_LOCATION);

  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
  getProfileBLAS().TPSTOP(QUDA_PROFILE_TOTAL);
  saveTuneCache();
}
