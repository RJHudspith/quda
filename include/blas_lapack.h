#include <quda_internal.h>

#pragma once

// assumes alpha = beta = 1
#define FLOPS_SGEMM(m, n, k) (2 * (m) * (n) * (k))
#define FLOPS_CGEMM(m, n, k) (8 * (m) * (n) * (k))

#define FMULS_GETRF(m_, n_)                                                                                            \
  (((m_) < (n_)) ? (0.5 * (m_) * ((m_) * ((n_) - (1. / 3.) * (m_)-1.) + (n_)) + (2. / 3.) * (m_)) :                    \
                   (0.5 * (n_) * ((n_) * ((m_) - (1. / 3.) * (n_)-1.) + (m_)) + (2. / 3.) * (n_)))
#define FADDS_GETRF(m_, n_)                                                                                            \
  (((m_) < (n_)) ? (0.5 * (m_) * ((m_) * ((n_) - (1. / 3.) * (m_)) - (n_)) + (1. / 6.) * (m_)) :                       \
                   (0.5 * (n_) * ((n_) * ((m_) - (1. / 3.) * (n_)) - (m_)) + (1. / 6.) * (n_)))

#define FLOPS_ZGETRF(m_, n_)                                                                                           \
  (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)))
#define FLOPS_CGETRF(m_, n_)                                                                                           \
  (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)))

#define FMULS_GETRI(n_) ((n_) * ((5. / 6.) + (n_) * ((2. / 3.) * (n_) + 0.5)))
#define FADDS_GETRI(n_) ((n_) * ((5. / 6.) + (n_) * ((2. / 3.) * (n_)-1.5)))

#define FLOPS_ZGETRI(n_) (6. * FMULS_GETRI((double)(n_)) + 2.0 * FADDS_GETRI((double)(n_)))
#define FLOPS_CGETRI(n_) (6. * FMULS_GETRI((double)(n_)) + 2.0 * FADDS_GETRI((double)(n_)))

namespace quda
{
  namespace blas_lapack
  {
    // just a little utility for checks to avoid repetition                                                                                                   
    static inline void testmaxblas( const char *str , const int a , const int b )                                                                             
    {                                                                                                                                                         
      if( a < std::max(1,b) ) {                                                                                                                               
	errorQuda("%s=%d must be >= max(1,%d)", str , a , b );                                                                                                
      }                                                                                                                                                       
    }
    // run some simple tests erroring if we catch anything                                                                                                    
    static inline void runBLASchecks( const QudaBLASParam blas_param )
    {
      // Sanity checks on parameters - check that min_dim is not zero as that doesn't make sense                                                              
      const int min_dim = std::min(blas_param.m, std::min(blas_param.n, blas_param.k));
      if (min_dim <= 0) {
	errorQuda("BLAS dims must be positive: m=%d, n=%d, k=%d", blas_param.m, blas_param.n, blas_param.k);
      }
      // error if any stride is negative                                                                                                                      
      const int min_stride = std::min(std::min(blas_param.a_stride, blas_param.b_stride), blas_param.c_stride);
      if (min_stride < 0) {
	errorQuda("BLAS strides must be positive or zero: a_stride=%d, b_stride=%d, c_stride=%d", blas_param.a_stride,
		  blas_param.b_stride, blas_param.c_stride);
      }
      // error if the batch value is non-positve                                                                                                              
      if (blas_param.batch_count < 1) { errorQuda("Batches must be greater than 0: batches=%d", blas_param.batch_count); }
      // Leading dims are dependendent on the matrix op type.                                                                                                 
      if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
	if (blas_param.trans_a == QUDA_BLAS_OP_N) { testmaxblas( "lda" , blas_param.lda , blas_param.m ) ;
	} else {                                    testmaxblas( "lda" , blas_param.lda , blas_param.k ) ; }
	if (blas_param.trans_b == QUDA_BLAS_OP_N) { testmaxblas( "ldb" , blas_param.ldb , blas_param.k ) ;
	} else {                                    testmaxblas( "ldb" , blas_param.ldb , blas_param.n ) ; }
	testmaxblas( "ldc" , blas_param.ldc , blas_param.m ) ;
      } else {
	// rowmajor tests and a swap                                                                                                                          
	if( blas_param.trans_a == QUDA_BLAS_OP_N) { testmaxblas( "lda" , blas_param.lda , blas_param.k ) ;
	} else {                                    testmaxblas( "lda" , blas_param.lda , blas_param.m ) ; }
	if (blas_param.trans_b == QUDA_BLAS_OP_N) { testmaxblas( "ldb" , blas_param.ldb , blas_param.n ) ;
	} else {                                    testmaxblas( "ldb" , blas_param.ldb , blas_param.k ) ; }
	testmaxblas( "ldc" , blas_param.ldc , blas_param.n ) ;
      }
    }

    bool use_native();
    void set_native(bool native);

    /**
       The native namespace is where we can deploy target specific
       blas/lapack operations, using vendor-specific libraries.  In
       the case of CUDA, this corresponds to the use of cuBLAS.
     */
    namespace native
    {

      /**
         @brief Create the BLAS context
      */
      void init();

      /**
         @brief Destroy the BLAS context
      */
      void destroy();

      /**
         @brief Batch inversion the matrix field using an LU decomposition method.
         @param[out] Ainv Matrix field containing the inverse matrices
         @param[in] A Matrix field containing the input matrices
         @param[in] n Dimension each matrix
         @param[in] batch Problem batch size
         @param[in] precision Precision of the input/output data
         @param[in] Location of the input/output data
         @return Number of flops done in this computation
      */
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision precision,
                                  QudaFieldLocation location);

      /**
         @brief Strided Batch GEMM. This function performs N GEMM type operations in a
         strided batched fashion transparently interfacing with cuBLAS or hipBLAS

	 Consider the row-major (non-transposed) batched BLAS routine, the code here does
	 for p in blas_param.batch_count
	   for m in blas_param.m 
	     for n in blas_param.n
	       for k in blas_param.k
	         sum += A[k+m*blas_param.lda+p*blas_param.stride_a]*B[n+k*blas_param.ldc+p*blas_param.stride_b]
	       C[n+m*ldc+p*blas_param.c_stride] = sum
	 If the user gives blas_param.batch_count = 1 this is obviously just a *GEMM

         @param[in] A Matrix field containing the A input matrices
         @param[in] B Matrix field containing the B input matrices
         @param[in/out] C Matrix field containing the result, and matrix to be added
         @param[in] cublas_param Parameter structure defining the GEMM type
         @param[in] Location of the input/output data
         @return Number of flops done in this computation
      */
      long long stridedBatchGEMM(const void *A, const void *B, void *C, QudaBLASParam blas_param, const QudaFieldLocation location);

    } // namespace native

    /**
       The generic namespace is where we can deploy any
       target-independent blas/lapack operations that are not supported
       on the native target.  To this end, we use Eigen on the host.
     */
    namespace generic
    {
      /**
         @brief Create the BLAS context
      */
      void init();

      /**
         @brief Destroy the BLAS context
      */
      void destroy();

      /**
         @brief Batch inversion the matrix field using an LU decomposition method.
         @param[out] Ainv Matrix field containing the inverse matrices
         @param[in] A Matrix field containing the input matrices
         @param[in] n Dimension each matrix
         @param[in] batch Problem batch size
         @param[in] precision Precision of the input/output data
         @param[in] Location of the input/output data
         @return Number of flops done in this computation
      */
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision precision,
                                  QudaFieldLocation location);

      /**
         @brief Strided Batch GEMM. This function performs N GEMM type operations in a
         strided batched fashion
         @param[in] A Matrix field containing the A input matrices
         @param[in] B Matrix field containing the B input matrices
         @param[in/out] C Matrix field containing the result, and matrix to be added
         @param[in] blas_param Parameter structure defining the GEMM type
         @param[in] Location of the input/output data
         @return Number of flops done in this computation
      */
      long long stridedBatchGEMM(const void *A, const void *B, void *C, QudaBLASParam blas_param, const QudaFieldLocation location);

    } // namespace generic
  }   // namespace blas_lapack
} // namespace quda
