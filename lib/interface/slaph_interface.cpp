#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <blas_quda.h>
#include <tune_quda.h>

#include <color_spinor_field.h>
#include <contract_quda.h>

using namespace quda;

TimeProfile &getProfileBaryonKernel();
TimeProfile &getProfileBaryonKernelModeTripletsA();
TimeProfile &getProfileBaryonKernelModeTripletsB();
TimeProfile &getProfileAccumulateEvecs();
TimeProfile &getProfileColorContract();
TimeProfile &getProfileColorCross();
TimeProfile &getProfileBLAS();
TimeProfile &getProfileCurrentKernel();

static const double OneGB = 1024.*1024.*1024.;

// copy Fourier twiddles to the device
static inline void
device_hostmom( const double _Complex *host_mom ,
		void *d_mom ,
		const size_t size ,
		const int precision )
{
  if( precision == QUDA_SINGLE_PRECISION ) {
    float _Complex *tmp = (float _Complex*)calloc( size , sizeof( float _Complex ) ) ;
    for( size_t i = 0 ; i < size ; i++ ) {
      tmp[i] = (float _Complex)host_mom[i] ;
    }
    qudaMemcpy(d_mom, tmp , size*2*precision, qudaMemcpyHostToDevice);  
    free( tmp ) ;
  } else {
    qudaMemcpy(d_mom, host_mom, size*2*precision, qudaMemcpyHostToDevice);  
  }
}

static inline void
hostreturn( const void *d_ret ,
	    double _Complex *return_array ,
	    const size_t size ,
	    const int precision )
{
  if( precision == QUDA_SINGLE_PRECISION ) {
    float _Complex *tmp = (float _Complex*)calloc( size , sizeof( float _Complex ) ) ;
    qudaMemcpy(tmp, d_ret, size*2*precision, qudaMemcpyDeviceToHost);
    for( size_t i = 0 ; i < size ; i++ ) {
      return_array[i] = (double _Complex)tmp[i] ;
    }
    free( tmp ) ;
  } else {
    qudaMemcpy(return_array, d_ret, size*2*precision, qudaMemcpyDeviceToHost);  
  }
}

// device-side
static void
apply_noises( const std::vector<ColorSpinorField> &evec ,
	      const ColorSpinorParam cuda_evec_param ,
	      std::vector<ColorSpinorField> &quda_q1 ,
	      const std::vector<std::complex<double>> &coeffs1 ,
	      std::vector<ColorSpinorField> &quda_q2 ,
	      const std::vector<std::complex<double>> &coeffs2 ,
	      std::vector<ColorSpinorField> &quda_q3 ,
	      const std::vector<std::complex<double>> &coeffs3 )
{
  const size_t n1 = quda_q1.size() , n2 = quda_q2.size() , n3 = quda_q3.size() ;
  const size_t nEv = evec.size() ;
  std::vector<ColorSpinorField> quda_evec(1) ;
  quda_evec[0] = ColorSpinorField(cuda_evec_param);
  for (size_t i=0; i<nEv; i++) {
    quda_evec[0] = evec[i] ;
    blas::block::caxpy( {coeffs1.begin()+n1*i,coeffs1.begin()+n1*i+n1},
			{quda_evec[0]}, {quda_q1.begin(),quda_q1.end()} ) ;
    blas::block::caxpy( {coeffs2.begin()+n2*i,coeffs2.begin()+n2*i+n2},
			{quda_evec[0]}, {quda_q2.begin(),quda_q2.end()} ) ;
    blas::block::caxpy( {coeffs3.begin()+n3*i,coeffs3.begin()+n3*i+n3},
			{quda_evec[0]}, {quda_q3.begin(),quda_q3.end()} ) ;
  }
}

void laphBaryonKernel( const int n1, const int n2, const int n3, const int nMom,
		       const double _Complex *host_coeffs1, 
		       const double _Complex *host_coeffs2, 
		       const double _Complex *host_coeffs3,
		       const double _Complex *host_mom, 
		       const int nEv,
		       void **host_evec,
		       QudaInvertParam inv_param,
		       double _Complex *return_array,
		       const int blockSizeMomProj,
		       const int X[4] )
{
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_TOTAL);
  // appropriate checks and balances
  if( sizeof(Complex) != sizeof(double _Complex) ) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( blockSizeMomProj > (n1*n2*n3) ) {
    errorQuda( "Block size mom proj %d > %d\n", blockSizeMomProj, n1*n2*n3 ) ;
  }
  if( inv_param.cuda_prec != QUDA_DOUBLE_PRECISION &&
      inv_param.cuda_prec != QUDA_SINGLE_PRECISION ) {
    errorQuda( "Unsupported device precision %d" , inv_param.cuda_prec ) ;
  }
  const size_t nSp    = X[0]*X[1]*X[2] ;
  const size_t nSites = nSp*X[3] ;
  const int precision = inv_param.cuda_prec ;
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_INIT);
  const lat_dim_t x = { X[0] , X[1] , X[2] , X[3] } ;
  ColorSpinorParam cpu_evec_param( host_evec, inv_param, x, false, QUDA_CPU_FIELD_LOCATION );
  cpu_evec_param.nSpin = 1;
  std::vector<ColorSpinorField> evec(nEv);
  for (int iEv=0; iEv<nEv; ++iEv) {
    cpu_evec_param.v = host_evec[iEv];
    evec[iEv] = ColorSpinorField(cpu_evec_param) ;
  }
  // evec parameters
  ColorSpinorParam cuda_evec_param( cpu_evec_param, inv_param, QUDA_CUDA_FIELD_LOCATION );
  cuda_evec_param.setPrecision( inv_param.cuda_prec, inv_param.cuda_prec, true );
  // Create q1, q2, and q3 temporaries
  ColorSpinorParam cuda_q1_param( cuda_evec_param, inv_param, QUDA_CUDA_FIELD_LOCATION );
  ColorSpinorParam cuda_q2_param( cuda_evec_param, inv_param, QUDA_CUDA_FIELD_LOCATION );
  ColorSpinorParam cuda_q3_param( cuda_evec_param, inv_param, QUDA_CUDA_FIELD_LOCATION );
  cuda_q1_param.create = cuda_q2_param.create = cuda_q3_param.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<std::complex<double>> coeffs1(n1*nEv), coeffs2(n2*nEv), coeffs3(n3*nEv) ;
  std::vector<ColorSpinorField> quda_q1(n1), quda_q2(n2), quda_q3(n3) ;
  for(int i=0; i<n1; i++) {
    quda_q1[i] = ColorSpinorField(cuda_q1_param) ;
    for( int j = 0 ; j < nEv ; j++ ) { coeffs1[j*n1+i] = (std::complex<double>)host_coeffs1[j+i*nEv] ; }
  }
  for(int i=0; i<n2; i++) {
    quda_q2[i] = ColorSpinorField(cuda_q2_param) ;
    for( int j = 0 ; j < nEv ; j++ ) { coeffs2[j*n2+i] = (std::complex<double>)host_coeffs2[j+i*nEv] ; }
  }
  for(int i=0; i<n3; i++) {
    quda_q3[i] = ColorSpinorField(cuda_q3_param) ;
    for( int j = 0 ; j < nEv ; j++ ) { coeffs3[j*n3+i] = (std::complex<double>)host_coeffs3[j+i*nEv] ; }
  }
  // device temporaries, momentum, and return buffers. All pretty small
  const size_t data_tmp_bytes = (size_t)blockSizeMomProj*(size_t)nSites*2*precision ;
  const size_t data_ret_bytes = (size_t)(X[3]*nMom)*(size_t)(n1*n2*n3)*2*precision ;
  const size_t data_mom_bytes = (size_t)(nMom*nSp)*2*precision ;
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  void *d_ret = pool_device_malloc(data_ret_bytes);
  void *d_mom = pool_device_malloc(data_mom_bytes);
  if( getVerbosity() >= QUDA_SUMMARIZE ) {
    printfQuda( "Tmp %f | ret %f | mom %f [GB]\n" ,
		data_tmp_bytes/OneGB ,
		data_ret_bytes/OneGB ,
		data_mom_bytes/OneGB ) ;
  }
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_INIT);  
  // Copy host_mom data to device
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_H2D);
  device_hostmom( host_mom , d_mom , nMom*nSp , precision ) ;
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_H2D);

  apply_noises( evec , cuda_evec_param ,
		quda_q1 , coeffs1 ,
		quda_q2 , coeffs2 ,
		quda_q3 , coeffs3 ) ;

  // usual momentum contraction, strided and blocked
  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom;
  cublas_param_mom_sum.n = X[3];
  cublas_param_mom_sum.k = nSp;
  cublas_param_mom_sum.lda = nSp;
  cublas_param_mom_sum.ldb = nSp;
  cublas_param_mom_sum.ldc = X[3]*n1*n2*n3;
  cublas_param_mom_sum.a_stride = 0 ;
  cublas_param_mom_sum.b_stride = nSites ;
  cublas_param_mom_sum.c_stride = X[3] ;
  cublas_param_mom_sum.batch_count = blockSizeMomProj;
  cublas_param_mom_sum.alpha = 1.0 ; cublas_param_mom_sum.beta = 0.0 ;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = (precision == QUDA_SINGLE_PRECISION) ? \
    QUDA_BLAS_DATATYPE_C : QUDA_BLAS_DATATYPE_Z;
  cublas_param_mom_sum.blas_type = QUDA_BLAS_GEMM ;
  // Create device diquark vector
  ColorSpinorParam cuda_diq_param( cuda_evec_param , inv_param , QUDA_CUDA_FIELD_LOCATION ) ;
  ColorSpinorField quda_diq( cuda_diq_param ) ;
  int nInBlock = 0 , blockStart = 0 ;
  for( int dil1=0; dil1<n1; dil1++ ) {
    for( int dil2=0; dil2<n2; dil2++ ) {
      getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(quda_q1[dil1], quda_q2[dil2], quda_diq);
      getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
      for (int dil3=0; dil3<n3; dil3++) {
	getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);	
	colorContractQuda(quda_diq, quda_q3[dil3], (char*)d_tmp + nSites*nInBlock*2*precision);
	getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;
	if (nInBlock == blockSizeMomProj ) {
	  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
	  blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, (char*)d_ret + X[3]*blockStart*2*precision,
						cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
	  blockStart += nInBlock ;
	  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
	  nInBlock = 0;
	}
      }
    }
  }
  // overspill is less efficient than exact division but more flexible
  if( nInBlock > 0 ) {
    getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
    cublas_param_mom_sum.batch_count = nInBlock;
    blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, (char*)d_ret + X[3]*blockStart*2*precision,
					  cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
  }
  // Copy return array back to host
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_D2H);
  hostreturn( d_ret , return_array , (size_t)(X[3]*nMom)*(size_t)(n1*n2*n3) , precision ) ;
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_D2H);
  // Clean up memory allocations
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  pool_device_free(d_ret);
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_FREE);
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_TOTAL);
}

// mode tripletA
void laphBaryonKernelComputeModeTripletA( const int nMom, const int nEv, const int blockSizeMomProj,
					  void **host_evec, 
					  const double _Complex *host_mom,
					  QudaInvertParam inv_param,
					  double _Complex *return_array,
					  const int X[4])
{
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_TOTAL);  
  const size_t nSp    = X[0]*X[1]*X[2];
  const size_t nSites = nSp*X[3] ;
  const size_t nEvChoose3 = nEv*(nEv-1)/2*(nEv-2)/3;
  // appropriate checks and balances
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( (size_t)blockSizeMomProj > nEvChoose3 ) {
    errorQuda("Block size mom proj %zu > %zu", (size_t)blockSizeMomProj, nEvChoose3);
  }
  if( inv_param.cuda_prec != QUDA_SINGLE_PRECISION &&
      inv_param.cuda_prec != QUDA_DOUBLE_PRECISION ) {
    errorQuda( "Unsupported device precision %d" , inv_param.cuda_prec ) ;
  }
  const int precision = inv_param.cuda_prec ;
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_INIT);
  // Parameter object describing evecs
  const lat_dim_t x = { X[0] , X[1] , X[2] , X[3] } ;
  ColorSpinorParam cpu_evec_param(host_evec, inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_evec_param.nSpin = 1;
  std::vector<ColorSpinorField> evec(nEv) ;
  for (int iEv=0; iEv<nEv; ++iEv) {
    cpu_evec_param.v = host_evec[iEv];
    evec[iEv] = ColorSpinorField(cpu_evec_param);
  }
  // chuck all the evecs on the GPU, this should probably be tiled - TODO
  ColorSpinorParam cuda_evec_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_evec_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField> quda_evec(nEv);
  for (int i=0; i<nEv; i++) {
    quda_evec[i] = ColorSpinorField(cuda_evec_param) ;
    quda_evec[i] = evec[i] ; // CPU -> GPU
  }
  // Device side temp array (complBuf in chroma_laph)
  const size_t data_tmp_bytes = blockSizeMomProj*nSites*2*precision ;
  const size_t data_ret_bytes = (size_t)nEvChoose3*nMom*X[3]*2*precision ;
  const size_t data_mom_bytes = (size_t)(nMom*nSp)*2*precision ;
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  void *d_ret = pool_device_malloc(data_ret_bytes);
  void *d_mom = pool_device_malloc(data_mom_bytes);
  if( getVerbosity() >= QUDA_SUMMARIZE ) {
    const size_t OneGB = 1024*1024*1024;
    const size_t total_bytes = data_tmp_bytes + data_ret_bytes + data_mom_bytes ;
    printfQuda("d_tmp %fGB | d_ret %fGB | d_mom %fGB | total = %fGB\n",
	       (double)data_tmp_bytes/(OneGB), (double)data_ret_bytes/(OneGB),
	       (double)data_mom_bytes/(OneGB), (double)total_bytes/(OneGB)); 
  }
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_INIT);
  // Copy host data to device
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_H2D);
  device_hostmom( host_mom , d_mom , nMom*nSp , precision ) ;
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_H2D);
  // idea here like always is to do several ev-blocks at once in a zgemm
  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom ;
  cublas_param_mom_sum.k = nSp ;
  cublas_param_mom_sum.n = X[3] ;
  cublas_param_mom_sum.lda = nSp ;
  cublas_param_mom_sum.ldb = nSp ;
  cublas_param_mom_sum.ldc = X[3]*nEvChoose3;
  cublas_param_mom_sum.a_stride = 0 ;
  cublas_param_mom_sum.b_stride = nSites ;
  cublas_param_mom_sum.c_stride = X[3] ;
  cublas_param_mom_sum.batch_count = blockSizeMomProj;
  cublas_param_mom_sum.alpha = 1. ; cublas_param_mom_sum.beta = 0. ;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = ( precision == QUDA_SINGLE_PRECISION ) ? \
    QUDA_BLAS_DATATYPE_C : QUDA_BLAS_DATATYPE_Z;
  // Create device diquark vector
  ColorSpinorParam cuda_diq_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField quda_diq(cuda_diq_param) ;
  int nInBlock = 0, blockStart = 0;
  for (int aEv=0; aEv<nEv; aEv++) {
    for (int bEv=aEv+1; bEv<nEv; bEv++) {
      getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(quda_evec[aEv], quda_evec[bEv], quda_diq);
      getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
      for (int cEv=bEv+1; cEv<nEv; cEv++) {
	getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);
	colorContractQuda(quda_diq, quda_evec[cEv],(char*)d_tmp + nSites*nInBlock*2*precision);
	getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;
	if (nInBlock == blockSizeMomProj) {
	  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);  
	  blas_lapack::native::stridedBatchGEMM( d_mom, d_tmp, (char*)d_ret+X[3]*blockStart*2*precision,
						 cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION );
	  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
	  blockStart += nInBlock;
	  nInBlock = 0;
	}
      }
    }
  }
  if( nInBlock > 0 ) {
    cublas_param_mom_sum.batch_count = nInBlock;
    getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);  
    blas_lapack::native::stridedBatchGEMM( d_mom, d_tmp, (char*)d_ret+X[3]*blockStart*2*precision,
					   cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION );
    getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);    
  }
  // Copy return array back to host
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_D2H);
  hostreturn( d_ret , return_array , (size_t)nEvChoose3*nMom*X[3] , precision ) ;
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_D2H);
  // Clean up memory allocations
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  pool_device_free(d_ret);
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_FREE);
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_TOTAL);
}

void laphBaryonKernelComputeModeTripletB( const int n1, const int n2, const int n3,
					  const int nMom, const int nEv,
					  const double _Complex *host_coeffs1, 
					  const double _Complex *host_coeffs2, 
					  const double _Complex *host_coeffs3,
					  const double _Complex *host_mode_trip_buf,
					  double _Complex *return_array)
{
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_INIT); 
  // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  const size_t data_coeffs1_bytes = n1*nEv*2*QUDA_DOUBLE_PRECISION;
  const size_t data_coeffs2_bytes = n2*nEv*2*QUDA_DOUBLE_PRECISION;
  const size_t data_coeffs3_bytes = n3*nEv*2*QUDA_DOUBLE_PRECISION;  
  const size_t data_q3_bytes      = nEv*nEv*n3*2*QUDA_DOUBLE_PRECISION;
  const size_t data_tmp_bytes     = std::max( nEv*nEv*nEv , nEv*n2*n3 )*2*QUDA_DOUBLE_PRECISION ;
  const size_t data_ret_bytes     = nMom*n1*n2*n3*2*QUDA_DOUBLE_PRECISION;
  const size_t total_bytes = data_tmp_bytes + data_q3_bytes + data_coeffs3_bytes
    +data_coeffs1_bytes+data_coeffs2_bytes+data_tmp_bytes+data_ret_bytes;
  // Allocate required memory
  void *d_coeffs1 = pool_device_malloc(data_coeffs1_bytes);
  void *d_coeffs2 = pool_device_malloc(data_coeffs2_bytes);
  void *d_coeffs3 = pool_device_malloc(data_coeffs3_bytes);
  void *d_ret     = pool_device_malloc(data_ret_bytes);  
  void *d_tmp     = pool_device_malloc(data_tmp_bytes);
  void *d_q3      = pool_device_malloc(data_q3_bytes);
  if (getVerbosity() >= QUDA_VERBOSE) {
    printfQuda("mtb %gGB | q3 %gGB | coeffs3 %gGB \n",
	       data_tmp_bytes/OneGB, data_q3_bytes/OneGB, data_coeffs3_bytes/OneGB);
    printfQuda( "coeffs1 %gGB | coeffs2 %gGB | ret %gGB | total %gGB\n",
		data_coeffs1_bytes/OneGB, data_coeffs2_bytes/OneGB,
		data_ret_bytes/OneGB, total_bytes/OneGB);
  }
  // ZGEMM INIT
  QudaBLASParam cublas_param_1 = newQudaBLASParam();
  cublas_param_1.trans_a = QUDA_BLAS_OP_N;
  cublas_param_1.trans_b = QUDA_BLAS_OP_T;
  cublas_param_1.m = nEv;
  cublas_param_1.n = n3;
  cublas_param_1.k = nEv;
  cublas_param_1.lda = nEv;
  cublas_param_1.ldb = nEv;
  cublas_param_1.ldc = n3;
  cublas_param_1.a_stride = nEv*nEv ;
  cublas_param_1.b_stride = 0 ;
  cublas_param_1.c_stride = n3*nEv ;
  cublas_param_1.batch_count = nEv;
  cublas_param_1.alpha = 1.0 ; cublas_param_1.beta = 0.0 ;
  cublas_param_1.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_1.data_type = QUDA_BLAS_DATATYPE_Z;

  QudaBLASParam cublas_param_2 = newQudaBLASParam();
  cublas_param_2.trans_a = cublas_param_2.trans_b = QUDA_BLAS_OP_N;
  cublas_param_2.m = n2;
  cublas_param_2.n = n3;
  cublas_param_2.k = nEv;
  cublas_param_2.lda = nEv;
  cublas_param_2.ldb = n3;
  cublas_param_2.ldc = n3;
  cublas_param_2.a_stride = 0 ;
  cublas_param_2.b_stride = n3*nEv ;
  cublas_param_2.c_stride = n2*n3 ;
  cublas_param_2.batch_count = nEv ;
  cublas_param_2.alpha = 1.0 ; cublas_param_2.beta = 0.0 ;
  cublas_param_2.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_2.data_type = QUDA_BLAS_DATATYPE_Z;
  
  QudaBLASParam cublas_param_3 = newQudaBLASParam();
  cublas_param_3.trans_a = cublas_param_3.trans_b = QUDA_BLAS_OP_N;
  cublas_param_3.m = n1;
  cublas_param_3.n = n3;
  cublas_param_3.k = nEv;
  cublas_param_3.lda = nEv;
  cublas_param_3.ldb = n2*n3;
  cublas_param_3.ldc = n2*n3;
  cublas_param_3.a_stride = 0 ;
  cublas_param_3.b_stride = n3 ;
  cublas_param_3.c_stride = n3 ;
  cublas_param_3.batch_count = n2 ;
  cublas_param_3.alpha = 1.0 ; cublas_param_3.beta = 0.0 ;
  cublas_param_3.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_3.data_type = QUDA_BLAS_DATATYPE_Z;
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_INIT);
  // flush this guy
  qudaMemset( d_tmp , 0 , data_tmp_bytes ) ;
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_H2D);  
  qudaMemcpy(d_coeffs1, host_coeffs1, data_coeffs1_bytes, qudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs2, host_coeffs2, data_coeffs2_bytes, qudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs3, host_coeffs3, data_coeffs3_bytes, qudaMemcpyHostToDevice);  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_H2D);
  // Compute ZGEMMs
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_COMPUTE);  
  for(int p=0; p<nMom; p++) {
    qudaMemcpy( d_tmp, (double _Complex*)host_mode_trip_buf+p*nEv*nEv*nEv ,
	        data_tmp_bytes, qudaMemcpyHostToDevice ) ;
    blas_lapack::native::stridedBatchGEMM( d_tmp, d_coeffs3, d_q3, cublas_param_1,
					   QUDA_CUDA_FIELD_LOCATION ) ;
    blas_lapack::native::stridedBatchGEMM( d_coeffs2, d_q3, d_tmp,
					   cublas_param_2, QUDA_CUDA_FIELD_LOCATION);
    blas_lapack::native::stridedBatchGEMM( (double _Complex*)d_coeffs1, d_tmp,
					   (double _Complex*)d_ret + p*n1*n2*n3 ,
					   cublas_param_3, QUDA_CUDA_FIELD_LOCATION);
  }
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_COMPUTE);
  // Copy return array to host
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy( return_array, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost );
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_D2H);
  // Clean up all remaining memory allocations
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_coeffs1);
  pool_device_free(d_coeffs2);
  pool_device_free(d_coeffs3);
  pool_device_free(d_tmp);
  pool_device_free(d_q3);
  pool_device_free(d_ret);  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_FREE);
  saveTuneCache();
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_TOTAL);
}

// new GPU interface with better behaviour
void laphCurrentKernel( const int n1, const int n2, const int nMom,
			const int blockSizeMomProj,
			void **host_quark,
			void **host_quark_bar,
			const double _Complex *host_mom,
			QudaInvertParam inv_param,
			double _Complex *return_array,
			const int X[4])
{
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_INIT);  
  // Check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( blockSizeMomProj > (n1*n2) ) {
    errorQuda("block_size_mom_proj %d > (n1*n2) %d" , blockSizeMomProj, n1*n2 ) ;
  }
  if( inv_param.cuda_prec != QUDA_DOUBLE_PRECISION &&
      inv_param.cuda_prec != QUDA_SINGLE_PRECISION ) {
    errorQuda("Unsupported device precision") ;
  }
  const QudaPrecision precision = inv_param.cuda_prec ;
  // Some common variables
  const size_t nSp = X[0]*X[1]*X[2];
  const size_t nSites = nSp*X[3];
  const lat_dim_t x = { X[0] , X[1] , X[2] , X[3] } ;
  // Create device vectors for quarks
  ColorSpinorParam cpu_quark_param(host_quark, inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_param.nSpin = 1;
  std::vector<ColorSpinorField> quark(n2) ;
  for( int dil2 = 0 ; dil2 < n2 ; dil2++ ) {
    cpu_quark_param.v = host_quark[dil2] ;
    quark[dil2] = ColorSpinorField(cpu_quark_param) ;
  }
  ColorSpinorParam cuda_quark_param(cpu_quark_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_quark_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  // Create device vectors for quark_bar
  ColorSpinorParam cpu_quark_bar_param(host_quark_bar, inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_bar_param.nSpin = 1;
  ColorSpinorParam cuda_quark_bar_param(cpu_quark_bar_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_quark_bar_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);  
  // Device array to hold the entire return array
  const size_t data_ret_bytes = nMom*X[3]*n1*n2*2*precision;
  const size_t data_tmp_bytes = nSites*blockSizeMomProj*2*precision;
  const size_t data_mom_bytes = nMom*nSp*2*precision;
  void *d_ret = pool_device_malloc(data_ret_bytes);
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  void *d_mom = pool_device_malloc(data_mom_bytes);
  // momentum contractions are a batched strided BLAS
  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom ;
  cublas_param_mom_sum.n = X[3] ;
  cublas_param_mom_sum.k   = nSp ;
  cublas_param_mom_sum.lda = nSp ;
  cublas_param_mom_sum.ldb = nSp ;
  cublas_param_mom_sum.ldc = X[3] ;
  cublas_param_mom_sum.a_stride = 0 ; // mom stays the same
  cublas_param_mom_sum.b_stride = nSp*X[3] ;
  cublas_param_mom_sum.c_stride = X[3]*nMom ;
  cublas_param_mom_sum.batch_count = blockSizeMomProj ;
  cublas_param_mom_sum.alpha = 1.0; cublas_param_mom_sum.beta = 0.0;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = (inv_param.cuda_prec == QUDA_SINGLE_PRECISION) ? \
    QUDA_BLAS_DATATYPE_C : QUDA_BLAS_DATATYPE_Z ;
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_INIT);
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_H2D);
  // Copy host data to device for q2, q1 is done as we want
  std::vector<ColorSpinorField> quda_quark(n2) ;
  for (int dil2=0; dil2<n2; dil2++) {
    quda_quark[dil2] = ColorSpinorField(cuda_quark_param) ;
    quda_quark[dil2] = quark[dil2] ;
  }
  device_hostmom( host_mom , d_mom , nMom*nSp , precision ) ;
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_H2D);
  // doing too much work here as color contract (di1,dil2) == (dil2,dil1)*
  // so we could do a conjugated GEMM on the lower diagonal for the mom proj - TODO
  int nInBlock = 0 , blockStart = 0 ;
  for (int dil1=0; dil1<n1; dil1++) {
    cpu_quark_bar_param.v = host_quark_bar[dil1] ;
    ColorSpinorField quark_bar(cpu_quark_bar_param) ;
    ColorSpinorField quda_quark_bar(cuda_quark_bar_param) ;
    quda_quark_bar = quark_bar ;
    // just block dil2
    for (int dil2=0; dil2<n2; dil2++){
      getProfileCurrentKernel().TPSTART(QUDA_PROFILE_COMPUTE);
      innerProductQuda( quda_quark_bar, quda_quark[dil2], (char*)d_tmp+nSites*nInBlock*2*precision );
      getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_COMPUTE);
      nInBlock++ ;
      if( nInBlock == blockSizeMomProj ) {
	getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
	blas_lapack::native::stridedBatchGEMM( d_mom, d_tmp, (char*)d_ret+blockStart*X[3]*nMom*2*precision,
					       cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
	blockStart += nInBlock ;
	nInBlock = 0 ;
	getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
      }
    }
  }
  if( nInBlock > 0 ) {
    cublas_param_mom_sum.batch_count = nInBlock ;
    blas_lapack::native::stridedBatchGEMM( d_mom, d_tmp, (char*)d_ret+blockStart*X[3]*nMom*2*precision,
					   cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
  }
  // Copy device data back to host
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_D2H);
  hostreturn( d_ret , return_array , nMom*X[3]*n1*n2 , precision ) ;
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_D2H);
  // Clean up memory allocations
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_ret);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_FREE);
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_TOTAL);
}
