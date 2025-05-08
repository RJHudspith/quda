#include <timer.h>
#include <blas_lapack.h>
#include <eigen_helper.h>

//#define _DEBUG

namespace quda
{
  namespace blas_lapack
  {
    // whether we are using the native blas-lapack library
    static bool native_blas_lapack = true;
    bool use_native() { return native_blas_lapack; }
    void set_native(bool native) { native_blas_lapack = native; }
    namespace generic
    {
      void init() {}
      void destroy() {}
      // Batched inversion ckecking
      //---------------------------------------------------
      template <typename EigenMatrix, typename Float>
      void invertEigen(std::complex<Float> *A_eig, std::complex<Float> *Ainv_eig, int n, uint64_t batch)
      {
        EigenMatrix res = EigenMatrix::Zero(n, n);
        EigenMatrix inv = EigenMatrix::Zero(n, n);
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
	    res(k, j) = A_eig[batch * n * n + j * n + k];
	  }
        }

        inv = res.inverse();

        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) { Ainv_eig[batch * n * n + j * n + k] = inv(k, j); }
        }

        // Check result:
#ifdef _DEBUG
        EigenMatrix unit = EigenMatrix::Identity(n, n);
        EigenMatrix prod = res * inv;
        Float L2norm = ((prod - unit).norm() / (n * n));
        printfQuda("Eigen: Norm of (A * Ainv - I) batch %lu = %e\n", batch, L2norm);
#endif
      }
      //---------------------------------------------------
      // Batched Inversions
      //---------------------------------------------------
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision prec,
                                  QudaFieldLocation location)
      {
        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("BatchInvertMatrix (generic - Eigen): Nc = %d, batch = %lu\n", n, batch);

        size_t size = 2 * n * n * batch * prec;
        void *A_h = (location == QUDA_CUDA_FIELD_LOCATION ? pool_pinned_malloc(size) : A);
        void *Ainv_h = (location == QUDA_CUDA_FIELD_LOCATION ? pool_pinned_malloc(size) : Ainv);
        if (location == QUDA_CUDA_FIELD_LOCATION) { qudaMemcpy(A_h, A, size, qudaMemcpyDeviceToHost); }

        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);

        if (prec == QUDA_SINGLE_PRECISION) {
          std::complex<float> *A_eig = (std::complex<float> *)A_h;
          std::complex<float> *Ainv_eig = (std::complex<float> *)Ainv_h;

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (uint64_t i = 0; i < batch; i++) { invertEigen<MatrixXcf, float>(A_eig, Ainv_eig, n, i); }
          flops += batch * FLOPS_CGETRF(n, n);
        } else if (prec == QUDA_DOUBLE_PRECISION) {
          std::complex<double> *A_eig = (std::complex<double> *)A_h;
          std::complex<double> *Ainv_eig = (std::complex<double> *)Ainv_h;

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (uint64_t i = 0; i < batch; i++) { invertEigen<MatrixXcd, double>(A_eig, Ainv_eig, n, i); }
          flops += batch * FLOPS_ZGETRF(n, n);
        } else {
          errorQuda("%s not implemented for precision = %d", __func__, prec);
        }

        gettimeofday(&stop, NULL);
        long dsh = stop.tv_sec - start.tv_sec;
        long dush = stop.tv_usec - start.tv_usec;
        double timeh = dsh + 0.000001 * dush;

        if (getVerbosity() >= QUDA_VERBOSE) {
          int threads = 1;
#ifdef _OPENMP
          threads = omp_get_num_threads();
#endif
          printfQuda("CPU: Batched matrix inversion completed in %f seconds using %d threads with GFLOPS = %f\n", timeh,
                     threads, 1e-9 * flops / timeh);
        }

        if (location == QUDA_CUDA_FIELD_LOCATION) {
          pool_pinned_free(Ainv_h);
          pool_pinned_free(A_h);
          qudaMemcpy((void *)Ainv, Ainv_h, size, qudaMemcpyHostToDevice);
        }

        return flops;
      }

      // Srided Batched GEMM helpers
      //--------------------------------------------------------------------------
      template <typename EigenMat, typename T>
      void fillArray(EigenMat &EigenArr, T *arr, int rows, int cols, int ld, bool fill_eigen)
      {
        int counter = 0;
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            if (fill_eigen)
              EigenArr(i, j) = arr[counter];
            else
              arr[counter] = EigenArr(i, j);
            counter++;
          }
          counter += (ld - cols);
        }
      }

      template <typename EigenMat, typename T>
      static void GEMM(const void *A_h, const void *B_h, void *C_h, const T alpha, const T beta, const QudaBLASParam blas_param)
      {
        T *A_ptr = (T *)A_h , *B_ptr = (T *)B_h , *C_ptr = (T *)C_h ;
        // Eigen objects to store data
        EigenMat Amat = EigenMat::Zero(blas_param.m, blas_param.k);
        EigenMat Bmat = EigenMat::Zero(blas_param.k, blas_param.n);
        EigenMat Cmat = EigenMat::Zero(blas_param.m, blas_param.n);
        for (int batch = 0; batch < blas_param.batch_count ; batch++ ) {
          // Populate Eigen objects
          fillArray<EigenMat, T>(Amat, A_ptr + blas_param.a_stride*batch, blas_param.m, blas_param.k, blas_param.lda, true);
          fillArray<EigenMat, T>(Bmat, B_ptr + blas_param.b_stride*batch, blas_param.k, blas_param.n, blas_param.ldb, true);
          fillArray<EigenMat, T>(Cmat, C_ptr + blas_param.c_stride*batch, blas_param.m, blas_param.n, blas_param.ldc, true);
          // Apply op(A) and op(B)
          switch (blas_param.trans_a) {
          case QUDA_BLAS_OP_T: Amat.transposeInPlace(); break;
          case QUDA_BLAS_OP_C: Amat.adjointInPlace(); break;
          case QUDA_BLAS_OP_N: break;
          default: errorQuda("Unknown blas op type %d", blas_param.trans_a);
          }
          switch (blas_param.trans_b) {
          case QUDA_BLAS_OP_T: Bmat.transposeInPlace(); break;
          case QUDA_BLAS_OP_C: Bmat.adjointInPlace(); break;
          case QUDA_BLAS_OP_N: break;
          default: errorQuda("Unknown blas op type %d", blas_param.trans_b);
          }
          // Perform GEMM using Eigen
          Cmat = alpha * Amat * Bmat + beta * Cmat;
          // Write back to the C array
          fillArray<EigenMat, T>(Cmat, C_ptr+blas_param.c_stride*batch, blas_param.m, blas_param.n, blas_param.ldc, false);
        }
      }

      // just a little utility for checks to avoid repetition       
      static void testmaxblas( const char *str , const int a , const int b )
      {
        if( a < std::max(1,b) ) {
          errorQuda("%s=%d must be >= max(1,%d)", str , a , b );
        }
      }
      
      //------------------------------------------------------
      // Strided Batched GEMM conforming to cuBlas parameters
      //------------------------------------------------------
      long long stridedBatchGEMM(const void *A_data, const void *B_data, void *C_data,
				 QudaBLASParam blas_param, const QudaFieldLocation location)
      {
	if( location != QUDA_CPU_FIELD_LOCATION ) {
          errorQuda("StridedBatchGemm lapack eigen expects fields only on host");
	}
        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);
	runBLASchecks( blas_param ) ;
        // Swap A and B if in column order
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
          std::swap(blas_param.m, blas_param.n);
          std::swap(blas_param.lda, blas_param.ldb);
          std::swap(blas_param.trans_a, blas_param.trans_b);
          std::swap(blas_param.a_stride, blas_param.b_stride);
          std::swap(A_data, B_data);
        }
	size_t data_size = 4 ;
	switch( blas_param.data_type ) {
	case QUDA_BLAS_DATATYPE_S : data_size = 4  ; break ;
	case QUDA_BLAS_DATATYPE_D : data_size = 8  ; break ;
	case QUDA_BLAS_DATATYPE_C : data_size = 8  ; break ;
	case QUDA_BLAS_DATATYPE_Z : data_size = 16 ; break ;
	default :
	  errorQuda("Unrecognized data type ^d\n" , blas_param.data_type ) ;
	  break ;
	}
	switch( blas_param.data_type ) {
	case QUDA_BLAS_DATATYPE_Z : {
          typedef std::complex<double> Z;
          const Z alpha = blas_param.alpha;
          const Z beta = blas_param.beta;
          GEMM<MatrixXcd, Z>(A_data, B_data, C_data, alpha, beta, blas_param);
          flops += blas_param.batch_count * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;
	case QUDA_BLAS_DATATYPE_C : {
          typedef std::complex<float> C;
          const C alpha = blas_param.alpha;
          const C beta  = blas_param.beta;
          GEMM<MatrixXcf, C>(A_data, B_data, C_data, alpha, beta, blas_param);
          flops += blas_param.batch_count * FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;
	case QUDA_BLAS_DATATYPE_D : {
          typedef double D;
          const D alpha = (D)(static_cast<std::complex<double>>(blas_param.alpha).real());
          const D beta  = (D)(static_cast<std::complex<double>>(blas_param.beta).real());
          GEMM<MatrixXd, D>(A_data, B_data, C_data, alpha, beta, blas_param);
          flops += blas_param.batch_count * FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;
	case QUDA_BLAS_DATATYPE_S : {
          typedef float S;
          const S alpha = (S)(static_cast<std::complex<float>>(blas_param.alpha).real());
          const S beta  = (S)(static_cast<std::complex<float>>(blas_param.beta).real());
          GEMM<MatrixXf, S>(A_data, B_data, C_data, alpha, beta, blas_param);
          flops += blas_param.batch_count * FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;	  
	default :
          errorQuda("blasGEMM type %d not implemented\n", blas_param.data_type);
        }
        // Restore the swap of data pointers
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_COL) {
          std::swap(A_data, B_data);
        }
        qudaDeviceSynchronize();
        gettimeofday(&stop, NULL);
        const long ds = stop.tv_sec - start.tv_sec , dus = stop.tv_usec - start.tv_usec;
        const double time = ds + 0.000001 * dus;
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("Batched matrix GEMM completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);
        return flops;
      }
    } // namespace generic
  }   // namespace blas_lapack
} // namespace quda
