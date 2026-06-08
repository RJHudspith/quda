#include <blas_lapack.h>
#include <timer.h>
#ifdef NATIVE_LAPACK_LIB
#include <cublas_v2.h>
#include <malloc_quda.h>
#endif

//#define _DEBUG

#ifdef _DEBUG
#include <eigen_helper.h>
#endif

namespace quda
{
  namespace blas_lapack
  {
    namespace native
    {
#ifdef NATIVE_LAPACK_LIB
      static cublasHandle_t handle;
#endif
      static bool cublas_init = false;

      void init()
      {
        if (!cublas_init) {
#ifdef NATIVE_LAPACK_LIB
          cublasStatus_t error = cublasCreate(&handle);
          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("cublasCreate failed with error %d", error);
          else
            printfQuda("cublasCreated successfully\n");
          cublas_init = true;
#endif
        }
      }

      void destroy()
      {
        if (cublas_init) {
#ifdef NATIVE_LAPACK_LIB
          cublasStatus_t error = cublasDestroy(handle);
          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in destroying cublas context, error code = %d\n", error);
          cublas_init = false;
#endif
        }
      }

#ifdef _DEBUG
      template <typename EigenMatrix, typename Float>
      __host__ void checkEigen(std::complex<Float> *A_h, std::complex<Float> *Ainv_h, int n, uint64_t batch)
      {
        EigenMatrix A = EigenMatrix::Zero(n, n);
        EigenMatrix Ainv = EigenMatrix::Zero(n, n);
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            A(k, j) = A_h[batch * n * n + j * n + k];
            Ainv(k, j) = Ainv_h[batch * n * n + j * n + k];
          }
        }

        // Check result:
        EigenMatrix unit = EigenMatrix::Identity(n, n);
        EigenMatrix prod = A * Ainv;
        Float L2norm = ((prod - unit).norm() / (n * n));
        printfQuda("cuBLAS: Norm of (A * Ainv - I) batch %lu = %e\n", batch, L2norm);
      }
#endif

#ifdef NATIVE_LAPACK_LIB
      // FIXME do this in pipelined fashion to reduce memory overhead.
      long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision prec,
                                  QudaFieldLocation location)
      {
        init();
        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("BatchInvertMatrix (native - cuBLAS): Nc = %d, batch = %lu\n", n, batch);

        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);

        size_t size = 2 * n * n * prec * batch;
        void *A_d = location == QUDA_CUDA_FIELD_LOCATION ? A : pool_device_malloc(size);
        void *Ainv_d = location == QUDA_CUDA_FIELD_LOCATION ? Ainv : pool_device_malloc(size);
        if (location == QUDA_CPU_FIELD_LOCATION) qudaMemcpy(A_d, A, size, qudaMemcpyHostToDevice);

#ifdef _DEBUG
        // Debug code: Copy original A matrix to host
        if (prec == QUDA_SINGLE_PRECISION) {
          std::complex<float> *A_h
            = (location == QUDA_CUDA_FIELD_LOCATION ? static_cast<std::complex<float> *>(pool_host_pinned_malloc(size)) :
                                                      static_cast<std::complex<float> *>(A_d));
          if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy((void *)A_h, A_d, size, qudaMemcpyDeviceToHost);
        } else if (prec == QUDA_DOUBLE_PRECISION) {
          std::complex<double> *A_h
            = (location == QUDA_CUDA_FIELD_LOCATION ? static_cast<std::complex<double> *>(pool_host_pinned_malloc(size)) :
                                                      static_cast<std::complex<double> *>(A_d));
          if (location == QUDA_CUDA_FIELD_LOCATION) qudaMemcpy((void *)A_h, A_d, size, qudaMemcpyDeviceToHost);
        } else {
          errorQuda("%s not implemented for precision=%d", __func__, prec);
        }
#endif

        int *dipiv = static_cast<int *>(pool_device_malloc(batch * n * sizeof(int)));
        int *dinfo_array = static_cast<int *>(pool_device_malloc(batch * sizeof(int)));
        int *info_array = static_cast<int *>(pool_host_pinned_malloc(batch * sizeof(int)));
        memset(info_array, '0', batch * sizeof(int)); // silence memcheck warnings

        if (prec == QUDA_SINGLE_PRECISION) {
          typedef cuFloatComplex C;
          C **A_array = static_cast<C **>(pool_device_malloc(2 * batch * sizeof(C *)));
          C **Ainv_array = A_array + batch;
          C **A_array_h = static_cast<C **>(pool_host_pinned_malloc(2 * batch * sizeof(C *)));
          C **Ainv_array_h = A_array_h + batch;
          for (uint64_t i = 0; i < batch; i++) {
            A_array_h[i] = static_cast<C *>(A_d) + i * n * n;
            Ainv_array_h[i] = static_cast<C *>(Ainv_d) + i * n * n;
          }
          qudaMemcpy(A_array, A_array_h, 2 * batch * sizeof(C *), qudaMemcpyHostToDevice);

          cublasStatus_t error = cublasCgetrfBatched(handle, n, A_array, n, dipiv, dinfo_array, batch);
          flops += batch * FLOPS_CGETRF(n, n);

          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in LU decomposition (cublasCgetrfBatched), error code = %d\n", error);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), qudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          error = cublasCgetriBatched(handle, n, (const C **)A_array, n, dipiv, Ainv_array, n, dinfo_array, batch);
          flops += batch * FLOPS_CGETRI(n);

          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in matrix inversion (cublasCgetriBatched), error code = %d\n", error);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          pool_device_free(A_array);
          pool_host_pinned_free(A_array_h);

#ifdef _DEBUG
          // Debug code: Copy computed Ainv to host
          std::complex<float> *Ainv_h = static_cast<std::complex<float> *>(pool_host_pinned_malloc(size));
          qudaMemcpy((void *)Ainv_h, Ainv_d, size, qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) { checkEigen<MatrixXcf, float>(A_h, Ainv_h, n, i); }
          pool_host_pinned_free(Ainv_h);
          pool_host_pinned_free(A_h);
#endif
        } else if (prec == QUDA_DOUBLE_PRECISION) {
          typedef cuDoubleComplex Z;
          Z **A_array = static_cast<Z **>(pool_device_malloc(2 * batch * sizeof(Z *)));
          Z **Ainv_array = A_array + batch;
          Z **A_array_h = static_cast<Z **>(pool_host_pinned_malloc(2 * batch * sizeof(Z *)));
          Z **Ainv_array_h = A_array_h + batch;
          for (uint64_t i = 0; i < batch; i++) {
            A_array_h[i] = static_cast<Z *>(A_d) + i * n * n;
            Ainv_array_h[i] = static_cast<Z *>(Ainv_d) + i * n * n;
          }
          qudaMemcpy(A_array, A_array_h, 2 * batch * sizeof(Z *), qudaMemcpyHostToDevice);

          cublasStatus_t error = cublasZgetrfBatched(handle, n, A_array, n, dipiv, dinfo_array, batch);
          flops += batch * FLOPS_ZGETRF(n, n);

          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in LU decomposition (cublasZgetrfBatched), error code = %d\n", error);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), qudaMemcpyDeviceToHost);
          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          error = cublasZgetriBatched(handle, n, (const Z **)A_array, n, dipiv, Ainv_array, n, dinfo_array, batch);
          flops += batch * FLOPS_CGETRI(n);

          if (error != CUBLAS_STATUS_SUCCESS)
            errorQuda("\nError in matrix inversion (cublasCgetriBatched), error code = %d\n", error);

          qudaMemcpy(info_array, dinfo_array, batch * sizeof(int), qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) {
            if (info_array[i] < 0) {
              errorQuda("%lu argument had an illegal value or another error occured, such as memory allocation failed",
                        i);
            } else if (info_array[i] > 0) {
              errorQuda("%lu factorization completed but the factor U is exactly singular", i);
            }
          }

          pool_device_free(A_array);
          pool_host_pinned_free(A_array_h);

#ifdef _DEBUG
          // Debug code: Copy computed Ainv to host
          std::complex<double> *Ainv_h = static_cast<std::complex<double> *>(pool_host_pinned_malloc(size));
          qudaMemcpy((void *)Ainv_h, Ainv_d, size, qudaMemcpyDeviceToHost);

          for (uint64_t i = 0; i < batch; i++) { checkEigen<MatrixXcd, double>(A_h, Ainv_h, n, i); }
          pool_host_pinned_free(Ainv_h);
          pool_host_pinned_free(A_h);
#endif
        } else {
          errorQuda("%s not implemented for precision=%d", __func__, prec);
        }

        if (location == QUDA_CPU_FIELD_LOCATION) {
          qudaMemcpy(Ainv, Ainv_d, size, qudaMemcpyDeviceToHost);
          pool_device_free(Ainv_d);
          pool_device_free(A_d);
        }

        pool_device_free(dipiv);
        pool_device_free(dinfo_array);
        pool_host_pinned_free(info_array);

        qudaDeviceSynchronize();
        gettimeofday(&stop, NULL);
        long ds = stop.tv_sec - start.tv_sec;
        long dus = stop.tv_usec - start.tv_usec;
        double time = ds + 0.000001 * dus;

        if (getVerbosity() >= QUDA_VERBOSE)
          printfQuda("Batched matrix inversion completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);

        return flops;
      }
#else
      long long BatchInvertMatrix(void *, void *, const int, const uint64_t, QudaPrecision, QudaFieldLocation)
      {
        errorQuda("Native BLAS not built. Please build and use native BLAS or use generic BLAS");
        return 0; // Stops a compiler warning
      }
#endif

#ifdef NATIVE_LAPACK_LIB     
      // reverted to a more transparent interface with cublas
      long long stridedBatchGEMM( const void *A_data, const void *B_data, void *C_data, QudaBLASParam blas_param,
				  const QudaFieldLocation location)
      {
	if( location != QUDA_CUDA_FIELD_LOCATION ) {
          errorQuda("location of stridedBatchGEMM now must be just on the device");
	}
        long long flops = 0;
        timeval start, stop;
        gettimeofday(&start, NULL);
	// run the BLAS parameter sanity checks and do a swap if we aren't column-major as that is the native ordering
	runBLASchecks( blas_param ) ;
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_ROW) {
          std::swap(blas_param.m, blas_param.n);
          std::swap(blas_param.lda, blas_param.ldb);
          std::swap(blas_param.trans_a, blas_param.trans_b);
          std::swap(blas_param.a_stride, blas_param.b_stride);
          std::swap(A_data, B_data);
        }
        cublasOperation_t trans_a = CUBLAS_OP_N;
        switch (blas_param.trans_a) {
        case QUDA_BLAS_OP_N: trans_a = CUBLAS_OP_N; break;
        case QUDA_BLAS_OP_T: trans_a = CUBLAS_OP_T; break;
        case QUDA_BLAS_OP_C: trans_a = CUBLAS_OP_C; break;
        default: errorQuda("Unknown QUDA_BLAS_OP type %d\n", blas_param.trans_a);
        }
        cublasOperation_t trans_b = CUBLAS_OP_N;
        switch (blas_param.trans_b) {
        case QUDA_BLAS_OP_N: trans_b = CUBLAS_OP_N; break;
        case QUDA_BLAS_OP_T: trans_b = CUBLAS_OP_T; break;
        case QUDA_BLAS_OP_C: trans_b = CUBLAS_OP_C; break;
        default: errorQuda("Unknown QUDA_BLAS_OP type %d\n", blas_param.trans_b);
        }
	// who doesn't like switch statements?
	cublasStatus_t error ;
	switch( blas_param.data_type ) {
	case QUDA_BLAS_DATATYPE_Z : {
	  const cuDoubleComplex alpha = make_double2((double)(static_cast<std::complex<double>>(blas_param.alpha).real()),
						     (double)(static_cast<std::complex<double>>(blas_param.alpha).imag()));
	  const cuDoubleComplex beta  = make_double2((double)(static_cast<std::complex<double>>(blas_param.beta).real()),
						     (double)(static_cast<std::complex<double>>(blas_param.beta).imag()));
	  error = cublasZgemmStridedBatched(handle, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, &alpha,
					    (cuDoubleComplex*)A_data, blas_param.lda, blas_param.a_stride,
					    (cuDoubleComplex*)B_data, blas_param.ldb, blas_param.b_stride, &beta,
					    (cuDoubleComplex*)C_data, blas_param.ldc, blas_param.c_stride,
					    blas_param.batch_count) ;
	  if( error != CUBLAS_STATUS_SUCCESS ) errorQuda("\nError in cuBLASZGEMMStridedBatched, error code = %d\n", error);
	  flops += blas_param.batch_count*FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;
	case QUDA_BLAS_DATATYPE_C : {
	  const cuFloatComplex alpha = make_float2((float)(static_cast<std::complex<double>>(blas_param.alpha).real()),
						   (float)(static_cast<std::complex<double>>(blas_param.alpha).imag()));
	  const cuFloatComplex beta  = make_float2((float)(static_cast<std::complex<double>>(blas_param.beta).real()),
						   (float)(static_cast<std::complex<double>>(blas_param.beta).imag()));
	  error = cublasCgemmStridedBatched(handle, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, &alpha,
					    (cuFloatComplex*)A_data, blas_param.lda, blas_param.a_stride,
					    (cuFloatComplex*)B_data, blas_param.ldb, blas_param.b_stride, &beta,
					    (cuFloatComplex*)C_data, blas_param.ldc, blas_param.c_stride,
					    blas_param.batch_count) ;
	  if( error != CUBLAS_STATUS_SUCCESS ) errorQuda("\nError in cuBLASCGEMMStridedBatched, error code = %d\n", error);
          flops += blas_param.batch_count*FLOPS_CGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;
	case QUDA_BLAS_DATATYPE_D : {
	  const double alpha = (double)(static_cast<std::complex<double>>(blas_param.alpha).real());
	  const double beta  = (double)(static_cast<std::complex<double>>(blas_param.beta).real());
	  error = cublasDgemmStridedBatched(handle, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, &alpha,
					    (double*)A_data, blas_param.lda, blas_param.a_stride,
					    (double*)B_data, blas_param.ldb, blas_param.b_stride, &beta,
					    (double*)C_data, blas_param.ldc, blas_param.c_stride,
					    blas_param.batch_count) ;
	  if( error != CUBLAS_STATUS_SUCCESS ) errorQuda("\nError in cuBLASDGEMMStridedBatched, error code = %d\n", error);
          flops += blas_param.batch_count*FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;
	case QUDA_BLAS_DATATYPE_S : {
	  const float alpha = (float)(static_cast<std::complex<float>>(blas_param.alpha).real());
	  const float beta  = (float)(static_cast<std::complex<float>>(blas_param.beta).real());
	  error = cublasSgemmStridedBatched(handle, trans_a, trans_b, blas_param.m, blas_param.n, blas_param.k, &alpha,
					    (float*)A_data, blas_param.lda, blas_param.a_stride,
					    (float*)B_data, blas_param.ldb, blas_param.b_stride, &beta,
					    (float*)C_data, blas_param.ldc, blas_param.c_stride,
					    blas_param.batch_count) ;
	  if( error != CUBLAS_STATUS_SUCCESS) errorQuda("\nError in cuBLASSGEMMStridedBatched, error code = %d\n", error);
	  flops += blas_param.batch_count*FLOPS_SGEMM(blas_param.m, blas_param.n, blas_param.k);
	} break ;
	default :
          errorQuda("cublasGEMM type %d not implemented\n", blas_param.data_type);
	  break ;
	}
	// swap back the data pointers for A and B
        if (blas_param.data_order == QUDA_BLAS_DATAORDER_ROW) {
          std::swap(A_data, B_data);
        }
        qudaDeviceSynchronize();
        gettimeofday(&stop, NULL);
        const long ds = stop.tv_sec - start.tv_sec , dus = stop.tv_usec - start.tv_usec;
        const double time = ds + 0.000001 * dus;
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
          printfQuda("Batched matrix GEMM completed in %f seconds with GFLOPS = %f\n", time, 1e-9 * flops / time);
	}
        return flops;
      }
#else
      long long stridedBatchGEMM(const void *, const void *, void *, QudaBLASParam, const QudaFieldLocation)
      {
        errorQuda("Native BLAS not built. Please build and use native BLAS or use generic BLAS");
        return 0;
      }
#endif

    } // namespace native
  }   // namespace blas_lapack
} // namespace quda
