/**
   @blas_interface_test.cpp
   @brief tests the gemmBLASQuda interface using eigen comparing it to the device result

   An example call could be

   ./blas_interface_test --blas-data-type=D --blas-data-order=row --blas-gemm-mnk=2 1 6 --blas-gemm-leading-dims=6 4 4 --blas-gemm-strides=0 1 1 --blas-batch=4

   This does A_{2,6}xB_{6,4} -> C_{2,4} where the 4 is batched 4 times, so it really is just a matvec

 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex>

#include <inttypes.h>

#include <test.h>
#include <blas_reference.h>
#include <misc.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// if "--enable-testing true" is passed, we run the tests defined in here
#include <blas_interface_test_gtest.hpp>

static QudaBLASDataType blas_data_type = QUDA_BLAS_DATATYPE_Z;
static QudaBLASDataOrder blas_data_order = QUDA_BLAS_DATAORDER_ROW;
static QudaBLASType blas_test_type = QUDA_BLAS_GEMM;
static int blas_batch = 4;

static QudaBLASOperation blas_gemm_trans_a = QUDA_BLAS_OP_N;
static QudaBLASOperation blas_gemm_trans_b = QUDA_BLAS_OP_N;

// matrix multiply in 4 batches
static std::array<int, 3> blas_gemm_mnk = {4,4,4} ;
static std::array<int, 3> blas_gemm_leading_dims = {4,4,4};
static std::array<int, 3> blas_gemm_strides = {0,4*4,4*4};

static std::array<double, 2> blas_gemm_alpha_re_im = {1,0} ;
static std::array<double, 2> blas_gemm_beta_re_im = {0,0} ;
static int blas_lu_inv_mat_size = 128;

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info(QudaBLASParam &param)
{
  printfQuda("running the following test:\n");
  printfQuda("BLAS interface %s test\n", get_blas_type_str(param.blas_type));
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),dimPartitioned(3));
}

void setBLASParam(QudaBLASParam &blas_param)
{
  blas_param.trans_a = blas_gemm_trans_a;
  blas_param.trans_b = blas_gemm_trans_b;
  blas_param.m = blas_gemm_mnk[0];
  blas_param.n = blas_gemm_mnk[1];
  blas_param.k = blas_gemm_mnk[2];
  blas_param.lda = blas_gemm_leading_dims[0];
  blas_param.ldb = blas_gemm_leading_dims[1];
  blas_param.ldc = blas_gemm_leading_dims[2];
  blas_param.a_stride = blas_gemm_strides[0];
  blas_param.b_stride = blas_gemm_strides[1];
  blas_param.c_stride = blas_gemm_strides[2];
  memcpy(&blas_param.alpha, blas_gemm_alpha_re_im.data(), sizeof(__complex__ double));
  memcpy(&blas_param.beta, blas_gemm_beta_re_im.data(), sizeof(__complex__ double));
  blas_param.data_order = blas_data_order;
  blas_param.data_type = blas_data_type;
  blas_param.batch_count = blas_batch;
  blas_param.blas_type = blas_test_type;
  blas_param.inv_mat_size = blas_lu_inv_mat_size;
}

template <typename T>
static inline void fillR( void *A , const size_t arr_size )
{
  T *ptA = (T*)A ;
  for( size_t i = 0 ; i < arr_size ; i++ ) {
    ptA[i] = 2*(rand()/(T)RAND_MAX)-1 ;
  }
}

template <typename T>
static double getdev( const void *refC1 , const void *refC2 , const size_t arr_size )
{
  const T *pt1 = (const T*)refC1 , *pt2 = (const T*)refC2 ;
  double dev = 0. ;
  for( size_t i = 0 ; i < arr_size ; i++ ) {
    dev += (double)abs( pt1[i] - pt2[i] ) ;
  }
  return dev ;
}

double gemm_test(test_t test_param)
{
  QudaBLASParam blas_param = newQudaBLASParam();
  blas_data_type = ::testing::get<1>(test_param);
  blas_test_type = ::testing::get<0>(test_param);
  setBLASParam(blas_param);

  display_test_info(blas_param);

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
  // allocate
  void *refA  = pinned_malloc( A_bytes );
  void *refB  = pinned_malloc( B_bytes );

  void *refC1 = pinned_malloc( C_bytes );
  void *refC2 = pinned_malloc( C_bytes );
  memset( refC1 , 0. , C_bytes ) ;
  memset( refC2 , 0. , C_bytes ) ;

  switch( blas_param.data_type ) {
  case QUDA_BLAS_DATATYPE_S :
    fillR<float>( refA , arrayA_size ) ;
    fillR<float>( refB , arrayB_size ) ;
    break ;
  case QUDA_BLAS_DATATYPE_D :
    fillR<double>( refA , arrayA_size ) ;
    fillR<double>( refB , arrayB_size ) ;
    break ;
  case QUDA_BLAS_DATATYPE_C :
    fillR<float>( refA , arrayA_size ) ;
    fillR<float>( (float*)refA+arrayA_size , arrayA_size ) ;
    fillR<float>( refB , arrayB_size ) ;
    fillR<float>( (float*)refB+arrayB_size , arrayB_size ) ;    
    break ;
  case QUDA_BLAS_DATATYPE_Z :
    fillR<double>( refA , arrayA_size ) ;
    fillR<double>( (double*)refA+arrayA_size , arrayA_size ) ;
    fillR<double>( refB , arrayB_size ) ;
    fillR<double>( (double*)refB+arrayB_size , arrayB_size ) ;    
    break ;
  }

  // ok finally call the functions
  blasGEMMQuda( refA , refB , refC1 , QUDA_BOOLEAN_TRUE  , blas_param ) ;
  blasGEMMQuda( refA , refB , refC2 , QUDA_BOOLEAN_FALSE , blas_param ) ;

  // compute deviation
  double deviation = 12345 ;
  switch( blas_param.data_type ) {
  case QUDA_BLAS_DATATYPE_S :
    deviation = getdev<float>( refC1 , refC2 , arrayC_size ) ;
    break ;
  case QUDA_BLAS_DATATYPE_D :
    deviation = getdev<double>( refC1 , refC2 , arrayC_size ) ;
    break ;
  case QUDA_BLAS_DATATYPE_C :
    deviation = getdev<std::complex<float>>( refC1 , refC2 , arrayC_size ) ;
    break ;
  case QUDA_BLAS_DATATYPE_Z :
    deviation = getdev<std::complex<double>>( refC1 , refC2 , arrayC_size ) ;
    break ;
  }
  
  printfQuda( "Deviation (eigen to GPU) :: %e\n" , deviation ) ;

  host_free( refA ) ;
  host_free( refB ) ;
  host_free( refC1 ) ;
  host_free( refC2 ) ;
  
  return deviation ;
}

double lu_inv_test(test_t test_param)
{
  QudaBLASParam blas_param = newQudaBLASParam();
  blas_data_type = ::testing::get<1>(test_param);
  blas_test_type = ::testing::get<0>(test_param);
  setBLASParam(blas_param);

  display_test_info(blas_param);

  // Sanity checks on parameters
  //-------------------------------------------------------------------------

  // If the batch value is non-positve, we error out
  if (blas_param.batch_count <= 0) { errorQuda("Batches must be positive: batches=%d", blas_param.batch_count); }
  //-------------------------------------------------------------------------

  // Reference data is always in complex double
  size_t data_in_size = sizeof(double);

  int batches = blas_param.batch_count;
  uint64_t array_size = blas_param.inv_mat_size * blas_param.inv_mat_size;

  // Create host data reference arrays
  void *ref_array = pinned_malloc(batches * array_size * 2 * data_in_size);
  void *ref_array_inv = pinned_malloc(batches * array_size * 2 * data_in_size);
  prepare_ref_array(ref_array, batches, array_size, data_in_size, blas_data_type);

  // Create device array appropriate for the requested problem.
  void *dev_array = nullptr;
  void *dev_array_inv = nullptr;
  size_t data_out_size = 0;
  // For now, data is always complex for LU inversion.
  int re_im = 2;

  switch (blas_data_type) {
  case QUDA_BLAS_DATATYPE_C: data_out_size = sizeof(float); break;
  case QUDA_BLAS_DATATYPE_Z: data_out_size = sizeof(double); break;
  case QUDA_BLAS_DATATYPE_S:
  case QUDA_BLAS_DATATYPE_D:
  default: errorQuda("Unsupported data type %d\n", blas_data_type);
  }

  dev_array = pinned_malloc(batches * array_size * re_im * data_out_size);
  dev_array_inv = pinned_malloc(batches * array_size * re_im * data_out_size);

  copy_array(dev_array, ref_array, batches, array_size, data_out_size, blas_data_type);

  // Perform device LU inversion
  blasLUInvQuda(dev_array_inv, dev_array, native_blas_lapack ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE, &blas_param);

  double deviation = 0.0;
  if (verify_results) { deviation = blasLUInvQudaVerify(ref_array, dev_array_inv, array_size, &blas_param); }

  host_free(ref_array);
  host_free(ref_array_inv);
  host_free(dev_array);
  host_free(dev_array_inv);

  return deviation;
}

struct blas_interface_test : quda_test {

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);

    CLI::TransformPairs<QudaBLASDataType> blas_dt_map {
      {"C", QUDA_BLAS_DATATYPE_C}, {"Z", QUDA_BLAS_DATATYPE_Z}, {"S", QUDA_BLAS_DATATYPE_S}, {"D", QUDA_BLAS_DATATYPE_D}};

    CLI::TransformPairs<QudaBLASDataOrder> blas_data_order_map {{"row", QUDA_BLAS_DATAORDER_ROW},
                                                                {"col", QUDA_BLAS_DATAORDER_COL}};
    CLI::TransformPairs<QudaBLASOperation> blas_op_map {
      {"N", QUDA_BLAS_OP_N}, {"T", QUDA_BLAS_OP_T}, {"C", QUDA_BLAS_OP_C}};

    CLI::TransformPairs<QudaBLASType> blas_type_map {{"gemm", QUDA_BLAS_GEMM}, {"lu-inv", QUDA_BLAS_LU_INV}};

    // Option group for BLAS test related options
    auto opgroup = app->add_option_group("BLAS Interface", "Options controlling BLAS interface tests");

    opgroup
      ->add_option("--blas-data-type", blas_data_type,
                   "Whether to use single(S), double(D), and/or complex(C/Z) data types (default C)")
      ->transform(CLI::QUDACheckedTransformer(blas_dt_map));

    opgroup
      ->add_option("--blas-test-type", blas_test_type,
                   "Whether to perform the GEMM test or LU Inversion test (default GEMM)")
      ->transform(CLI::QUDACheckedTransformer(blas_type_map));

    opgroup
      ->add_option("--blas-data-order", blas_data_order,
                   "Whether data is in row major or column major order (default row)")
      ->transform(CLI::QUDACheckedTransformer(blas_data_order_map));

    opgroup
      ->add_option(
        "--blas-gemm-trans-a", blas_gemm_trans_a,
        "Whether to leave the A GEMM matrix as is (N), to transpose (T) or transpose conjugate (C) (default N) ")
      ->transform(CLI::QUDACheckedTransformer(blas_op_map));

    opgroup
      ->add_option(
        "--blas-gemm-trans-b", blas_gemm_trans_b,
        "Whether to leave the B GEMM matrix as is (N), to transpose (T) or transpose conjugate (C) (default N) ")
      ->transform(CLI::QUDACheckedTransformer(blas_op_map));

    opgroup
      ->add_option("--blas-gemm-alpha", blas_gemm_alpha_re_im,
                   "Set the complex value of alpha for GEMM (default {1.0,0.0}")
      ->expected(2);

    opgroup
      ->add_option("--blas-gemm-beta", blas_gemm_beta_re_im, "Set the complex value of beta for GEMM (default {1.0,0.0}")
      ->expected(2);

    opgroup
      ->add_option("--blas-gemm-mnk", blas_gemm_mnk,
                   "Set the dimensions of the A, B, and C matrices GEMM (default 128 128 128)")
      ->expected(3);

    opgroup
      ->add_option("--blas-gemm-leading-dims", blas_gemm_leading_dims,
                   "Set the leading dimensions A, B, and C matrices GEMM (default 128 128 128) ")
      ->expected(3);

    opgroup
      ->add_option("--blas-gemm-strides", blas_gemm_strides,
                   "Set the strides for GEMM matrices A, B, and C (default 1 1 1)")
      ->expected(3);

    opgroup->add_option("--blas-batch", blas_batch, "Set the number of batches for GEMM or LU inversion (default 16)");

    opgroup->add_option("--blas-lu-inv-mat-size", blas_lu_inv_mat_size,
                        "Set the size of the square matrix to invert via LU (default 128)");
  }

  blas_interface_test(int argc, char **argv) : quda_test("BLAS Interface Test", argc, argv) { }
};

int main(int argc, char **argv)
{
  blas_interface_test test(argc, argv);
  test.init();

  int result = 0;
  if (enable_testing) {
    result = test.execute();
    if (result) warningQuda("Google tests for QUDA BLAS failed.");
  } else {
    // Perform the BLAS op specified by the command line
    switch (blas_test_type) {
    case QUDA_BLAS_GEMM: {
      switch (blas_data_type) {
      case QUDA_BLAS_DATATYPE_S: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_S}); break;
      case QUDA_BLAS_DATATYPE_D: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_D}); break;
      case QUDA_BLAS_DATATYPE_C: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_C}); break;
      case QUDA_BLAS_DATATYPE_Z: gemm_test(test_t {QUDA_BLAS_GEMM, QUDA_BLAS_DATATYPE_Z}); break;
      default: errorQuda("Undefined QUDA BLAS data type %d\n", blas_data_type);
      }
      break;
    }
    case QUDA_BLAS_LU_INV: {
      switch (blas_data_type) {
      case QUDA_BLAS_DATATYPE_C: lu_inv_test(test_t {QUDA_BLAS_LU_INV, QUDA_BLAS_DATATYPE_C}); break;
      case QUDA_BLAS_DATATYPE_Z: lu_inv_test(test_t {QUDA_BLAS_LU_INV, QUDA_BLAS_DATATYPE_Z}); break;
      default: errorQuda("QUDA BLAS data type %d not supported for LU Inversion\n", blas_data_type);
      }
    } break;
    default: errorQuda("Unknown QUDA BLAS test type %d\n", blas_test_type);
    }
  }

  return result;
}
