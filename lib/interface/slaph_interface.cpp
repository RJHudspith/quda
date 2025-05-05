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

void laphBaryonKernel( const int n1, const int n2, const int n3, const int nMom,
		       const double _Complex *host_coeffs1, 
		       const double _Complex *host_coeffs2, 
		       const double _Complex *host_coeffs3,
		       const double _Complex *host_mom, 
		       const int nEv,
		       void **host_evec,
		       QudaInvertParam inv_param,
		       void *return_array,
		       const int blockSizeMomProj,
		       const int X[4] )
{
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_TOTAL);
  
  // checks and balances
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( (n1*n2*n3)%blockSizeMomProj != 0 ) {
    errorQuda( "Block size mom proj needs to divide %d %d\n" , n1*n2*n3 , blockSizeMomProj ) ;
  }
  const int nSites = X[0]*X[1]*X[2];

  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_INIT);
  lat_dim_t x = { X[0] , X[1] , X[2] , X[3] } ;
  ColorSpinorParam cpu_evec_param(host_evec, inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_evec_param.nSpin = 1;
  std::vector<ColorSpinorField> evec(nEv);
  for (int iEv=0; iEv<nEv; ++iEv) {
    cpu_evec_param.v = host_evec[iEv];
    evec[iEv] = ColorSpinorField(cpu_evec_param) ;
  }
  // Create device evecs
  ColorSpinorParam cuda_evec_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_evec_param.nSpin = 1;
  cuda_evec_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField*> quda_evec ;
  for (int i=0; i<nEv; i++) {
    quda_evec.push_back( ColorSpinorField::Create(cuda_evec_param) );
    *quda_evec[i] = evec[i] ; // load here because fuck it why not
  }
  // Create q1
  ColorSpinorParam cuda_q1_param(cuda_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_q1_param.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<Complex> coeffs1(n1*nEv) ;
  std::vector<ColorSpinorField*> quda_q1 ;
  for(int i=0; i<n1; i++) {
    quda_q1.push_back(ColorSpinorField::Create(cuda_q1_param));
    for( int j = 0 ; j < nEv ; j++ ) coeffs1[j*n1+i] = host_coeffs1[j+i*nEv] ;
  }
  // Create q2
  ColorSpinorParam cuda_q2_param(cuda_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_q2_param.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<Complex> coeffs2(n2*nEv) ;
  std::vector<ColorSpinorField*> quda_q2 ;
  for(int i=0; i<n2; i++) {
    quda_q2.push_back(ColorSpinorField::Create(cuda_q2_param));
    for( int j = 0 ; j < nEv ; j++ ) coeffs2[j*n2+i] = host_coeffs2[j+i*nEv] ;
  }
  // create q3
  ColorSpinorParam cuda_q3_param(cuda_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_q3_param.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<Complex> coeffs3(n3*nEv) ;
  std::vector<ColorSpinorField*> quda_q3 ;
  for(int i=0; i<n3; i++) {
    quda_q3.push_back(ColorSpinorField::Create(cuda_q3_param));
    for( int j = 0 ; j < nEv ; j++ ) coeffs3[j*n3+i] = host_coeffs3[j+i*nEv] ;
  }

  // device temporaries, momentum and return buffers
  const size_t data_tmp_bytes = blockSizeMomProj*X[0]*X[1]*X[2]*2*quda_q3[0]->Precision();
  const size_t data_ret_bytes = nMom*n1*n2*n3*2*quda_q3[0]->Precision();
  const size_t data_mom_bytes = nMom*nSites*2*quda_q3[0]->Precision();
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  void *d_ret = pool_device_malloc(data_ret_bytes);
  void *d_mom = pool_device_malloc(data_mom_bytes);
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_INIT);  

  // Copy host_mom data to device
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_H2D);
  qudaMemcpy(d_mom, host_mom, data_mom_bytes, qudaMemcpyHostToDevice);  
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_H2D);

  // Perfrom the caxpy to compute all q-vectors
  getProfileAccumulateEvecs().TPSTART(QUDA_PROFILE_COMPUTE);
  quda::blas::legacy::caxpy(coeffs1.data(), quda_evec , quda_q1 ) ;
  quda::blas::legacy::caxpy(coeffs2.data(), quda_evec , quda_q2 ) ;
  quda::blas::legacy::caxpy(coeffs3.data(), quda_evec , quda_q3 ) ;
  getProfileAccumulateEvecs().TPSTOP(QUDA_PROFILE_COMPUTE);

  // evecs irrelevant and can be purged here I gues if we are really desperate for space

  // Create device diquark vector
  ColorSpinorParam cuda_diq_param( cuda_evec_param , inv_param , QUDA_CUDA_FIELD_LOCATION ) ;
  ColorSpinorField quda_diq( cuda_diq_param ) ;

  // usual momentum contraction
  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom;
  cublas_param_mom_sum.n = 1 ;
  cublas_param_mom_sum.k = nSites;
  cublas_param_mom_sum.lda = nSites;
  cublas_param_mom_sum.ldb = nSites;
  cublas_param_mom_sum.ldc = n1*n2*n3;
  // stride it this time
  cublas_param_mom_sum.a_stride = 0 ;
  cublas_param_mom_sum.b_stride = nSites ;
  cublas_param_mom_sum.c_stride = 1 ; 
  cublas_param_mom_sum.batch_count = blockSizeMomProj;
  cublas_param_mom_sum.alpha = 1.0; cublas_param_mom_sum.beta = 0.0;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_BLAS_DATATYPE_Z;
  cublas_param_mom_sum.blas_type = QUDA_BLAS_GEMM ;
  
  int nInBlock = 0;
  for( int dil1=0; dil1<n1; dil1++ ) {
    for( int dil2=0; dil2<n2; dil2++ ) {
      getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(*quda_q1[dil1], *quda_q2[dil2], quda_diq);
      getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
      for (int dil3=0; dil3<n3; dil3++) {
	getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);	
	colorContractQuda(quda_diq, *quda_q3[dil3], (std::complex<double>*)d_tmp + nSites*nInBlock);
	getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;
	if (nInBlock == blockSizeMomProj ) {
	  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);	  
	  blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp,
						(std::complex<double>*)d_ret + (dil1*n2 + dil2)*n3 + dil3 - nInBlock + 1,
						cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
	  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);	  
	  nInBlock = 0;
	}
      }
    }
  }
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(return_array, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost);  
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_D2H);
  
  // Clean up memory allocations
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_FREE);
  // I know these are gross but I wanted to use the legacy caxpy
  for (int i=0; i<n1; i++ ) delete quda_q1[i];
  for (int i=0; i<n2; i++ ) delete quda_q2[i];
  for (int i=0; i<n3; i++ ) delete quda_q3[i];
  for (int i=0; i<nEv; i++) delete quda_evec[i];
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  pool_device_free(d_ret);
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_FREE);

  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_TOTAL);
}

// so I tied an onion to my belt, which was the style at the time
void laphBaryonKernelComputeModeTripletA( const int nMom, const int nEv, const int blockSizeMomProj,
					  void **host_evec, 
					  const double _Complex *host_mom,
					  QudaInvertParam inv_param,
					  double _Complex *return_array,
					  const int X[4])
{
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_TOTAL);
  
  // important that this only works on spatial nSites
  const size_t nSites = X[0]*X[1]*X[2];
  const size_t nEvChoose3 = nEv*(nEv-1)/2*(nEv-2)/3;  
  // appropriate checks and balances
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( nEvChoose3%blockSizeMomProj != 0 ) {
    errorQuda("Block size mom proj needs to divide %zu %d", nEvChoose3 , blockSizeMomProj);
  }
  
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

  // chuck all the evecs on the GPU
  ColorSpinorParam cuda_evec_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_evec_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField> quda_evec(nEv);
  for (int i=0; i<nEv; i++) {
    quda_evec[i] = ColorSpinorField(cuda_evec_param) ;
    quda_evec[i] = evec[i] ; // CPU -> GPU
  }
  
  // Create device diquark vector
  ColorSpinorParam cuda_diq_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField quda_diq(cuda_diq_param) ;
  
  // Device side temp array (complBuf in chroma_laph)
  const size_t data_tmp_bytes = blockSizeMomProj*X[0]*X[1]*X[2]*2*quda_evec[0].Precision();
  const size_t data_ret_bytes = nEvChoose3*nMom*2*quda_evec[0].Precision();
  const size_t data_mom_bytes = nMom*nSites*2*quda_evec[0].Precision();
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  void *d_ret = pool_device_malloc(data_ret_bytes);
  void *d_mom = pool_device_malloc(data_mom_bytes);
  if( getVerbosity() >= QUDA_SUMMARIZE ) {
    const size_t total_bytes = data_tmp_bytes+data_ret_bytes+data_mom_bytes ;
    printfQuda("d_tmp %fGB | d_ret %fGB | d_mom %fGB | total = %fGB\n",
	       data_tmp_bytes/OneGB, data_ret_bytes/OneGB,
	       data_mom_bytes/OneGB, total_bytes/OneGB); 
  }
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_INIT);

  // Copy host data to device
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_H2D);
  qudaMemcpy(d_mom, host_mom, data_mom_bytes, qudaMemcpyHostToDevice);  
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_H2D);

  // idea here like always is to do several ev-blocks at once in a zgemm
  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom;
  cublas_param_mom_sum.k = nSites;
  cublas_param_mom_sum.n = 1;
  cublas_param_mom_sum.lda = nSites;
  cublas_param_mom_sum.ldb = nSites;
  cublas_param_mom_sum.ldc = nEvChoose3;
  cublas_param_mom_sum.a_stride = 0;
  cublas_param_mom_sum.b_stride = nSites;
  cublas_param_mom_sum.c_stride = 1;
  cublas_param_mom_sum.batch_count = blockSizeMomProj;
  cublas_param_mom_sum.alpha = 1.0 ; cublas_param_mom_sum.beta = 0.0 ;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_BLAS_DATATYPE_Z;

  int nInBlock = 0, blockStart = 0;
  for (int aEv=0; aEv<nEv; aEv++) {
    for (int bEv=aEv+1; bEv<nEv; bEv++) {
      getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(quda_evec[aEv], quda_evec[bEv], quda_diq);
      getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
      for (int cEv=bEv+1; cEv<nEv; cEv++) {
	getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);
	colorContractQuda(quda_diq, quda_evec[cEv],(std::complex<double>*)d_tmp + nSites*nInBlock);
	getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;
	if (nInBlock == blockSizeMomProj) {
	  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);  
	  blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, (std::complex<double>*)d_ret+blockStart,
						cublas_param_mom_sum,
						QUDA_CUDA_FIELD_LOCATION);
	  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
	  blockStart += nInBlock;
	  nInBlock = 0;
	}
      }
    }
  }
  // Copy return array back to host  
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(return_array, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost);  
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_D2H);
  
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
  // number of EV indices (in first position) that this rank deals with
  const int nRanks = comm_size();  
  const int nSubEv = nEv / nRanks;
  const int iRank  = comm_rank();
  // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }  

  const size_t data_coeffs1_bytes = n1*nEv*2*QUDA_DOUBLE_PRECISION;
  const size_t data_coeffs2_bytes = n2*nEv*2*QUDA_DOUBLE_PRECISION;
  const size_t data_coeffs3_bytes = n3*nEv*2*QUDA_DOUBLE_PRECISION;  
  const size_t data_q3_bytes      = nSubEv*nEv*n3*2*QUDA_DOUBLE_PRECISION;
  const size_t data_tmp_bytes     = std::max( nSubEv*nEv*nEv , nSubEv*n2*n3 )*2*QUDA_DOUBLE_PRECISION ;
  const size_t data_ret_bytes     = nMom*n1*n2*n3*2*QUDA_DOUBLE_PRECISION;
  const size_t total_bytes = data_tmp_bytes + data_q3_bytes + data_coeffs3_bytes \
    +data_coeffs1_bytes + data_coeffs2_bytes + data_tmp_bytes + data_ret_bytes;

  // Allocate required memory
  void *d_tmp     = pool_device_malloc(data_tmp_bytes);
  void *d_q3      = pool_device_malloc(data_q3_bytes);
  void *d_coeffs1 = pool_device_malloc(data_coeffs1_bytes);
  void *d_coeffs2 = pool_device_malloc(data_coeffs2_bytes);
  void *d_coeffs3 = pool_device_malloc(data_coeffs3_bytes);
  void *d_ret     = pool_device_malloc(data_ret_bytes);  
  if (getVerbosity() >= QUDA_VERBOSE) {
    printfQuda("mtb %gGB | q3 %gGB | coeffs3 %gGB \n",
	       data_tmp_bytes/OneGB, data_q3_bytes/OneGB, data_coeffs3_bytes/OneGB,
	       "coeffs1 %gGB | coeffs2 %gGB | ret %gGB | total %gGB\n",
	       data_coeffs1_bytes/OneGB, data_coeffs2_bytes/OneGB,
	       data_ret_bytes/OneGB, total_bytes/OneGB);
  }

  // Initialise all the ZGEMMS
  QudaBLASParam cublas_param_1 = newQudaBLASParam();
  cublas_param_1.trans_a = QUDA_BLAS_OP_N;
  cublas_param_1.trans_b = QUDA_BLAS_OP_T;
  cublas_param_1.m = nSubEv*nEv;
  cublas_param_1.n = n3;
  cublas_param_1.k = nEv;
  cublas_param_1.lda = nEv;
  cublas_param_1.ldb = nEv;
  cublas_param_1.ldc = n3;
  cublas_param_1.batch_count = 1;
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
  cublas_param_2.c_stride = n2*nEv ;
  cublas_param_2.batch_count = nSubEv ;
  cublas_param_2.alpha = 1.0 ; cublas_param_2.beta = 0.0 ;
  cublas_param_2.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_2.data_type = QUDA_BLAS_DATATYPE_Z;

  QudaBLASParam cublas_param_3 = newQudaBLASParam();
  cublas_param_3.trans_a = cublas_param_3.trans_b = QUDA_BLAS_OP_N;
  cublas_param_3.m = n1;
  cublas_param_3.n = n2*n3;
  cublas_param_3.k = nSubEv;
  cublas_param_3.lda = nEv;
  cublas_param_3.ldb = n2*n3;
  cublas_param_3.ldc = n2*n3;
  cublas_param_3.batch_count = 1;
  cublas_param_3.alpha = 1.0 ; cublas_param_3.beta = 0.0 ;
  cublas_param_3.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_3.data_type = QUDA_BLAS_DATATYPE_Z;
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_INIT);

  // Copy coeffs to device
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_H2D);
  qudaMemcpy(d_coeffs3, host_coeffs3, data_coeffs3_bytes, qudaMemcpyHostToDevice);
  qudaMemcpy(d_coeffs1, host_coeffs1, data_coeffs1_bytes, qudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs2, host_coeffs2, data_coeffs2_bytes, qudaMemcpyHostToDevice);  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_H2D);
  
  // Compute ZGEMMs
  for(int p=0; p<nMom; p++) {
    getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_H2D);
    qudaMemcpy( d_tmp, (double _Complex*)host_mode_trip_buf+p*nSubEv*nEv*nEv ,
		data_tmp_bytes, qudaMemcpyHostToDevice ) ;
    getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_H2D);
    getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_COMPUTE);  
    blas_lapack::native::stridedBatchGEMM( d_tmp, d_coeffs3, d_q3, cublas_param_1,
                                           QUDA_CUDA_FIELD_LOCATION ) ;
    blas_lapack::native::stridedBatchGEMM( d_coeffs2, d_q3 , d_tmp ,
					   cublas_param_2, QUDA_CUDA_FIELD_LOCATION);
    blas_lapack::native::stridedBatchGEMM( (double _Complex*)d_coeffs1+iRank*nSubEv,
					   d_tmp,
					   (double _Complex*)d_ret + p*n1*n2*n3,
					   cublas_param_3, QUDA_CUDA_FIELD_LOCATION);
    getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_COMPUTE); 
  }
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(return_array, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost);  
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
			const int block_size_mom_proj,
			void **host_quark,
			void **host_quark_bar,
			const double _Complex *host_mom,
			QudaInvertParam inv_param,
			void *return_array,
			const int X[4])
{
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_TOTAL);

  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_INIT);  
  // Check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( (n2*n1)%block_size_mom_proj != 0 ) {
    errorQuda("I only support block sizes that are factors of n1*n2") ;
  }
  
  // Some common variables
  const size_t n_spatial_sites = X[0]*X[1]*X[2];
  const size_t n_sites = n_spatial_sites*X[3];
  const QudaPrecision precision = QUDA_DOUBLE_PRECISION;
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
  const size_t data_tmp_bytes = n_sites*2*precision*block_size_mom_proj;
  const size_t data_mom_bytes = nMom*n_spatial_sites*2*precision;
  void *d_ret = pool_device_malloc(data_ret_bytes);
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  void *d_mom = pool_device_malloc(data_mom_bytes);
  
  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom ;
  cublas_param_mom_sum.n = X[3] ;
  cublas_param_mom_sum.k   = n_spatial_sites ;
  cublas_param_mom_sum.lda = n_spatial_sites ;
  cublas_param_mom_sum.ldb = n_spatial_sites ;
  cublas_param_mom_sum.ldc = X[3] ;
  cublas_param_mom_sum.a_stride = X[3] ;
  cublas_param_mom_sum.b_stride = n_spatial_sites*X[3] ;
  cublas_param_mom_sum.c_stride = X[3]*nMom ;
  cublas_param_mom_sum.batch_count = block_size_mom_proj ;
  cublas_param_mom_sum.alpha = 1.0; cublas_param_mom_sum.beta = 0.0;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_BLAS_DATATYPE_Z;
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_INIT);
  
  // Copy host data to device
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_H2D);
  std::vector<ColorSpinorField> quda_quark(n2) ;
  for (int dil2=0; dil2<n2; dil2++) {
    quda_quark[dil2] = ColorSpinorField(cuda_quark_param) ;
    quda_quark[dil2] = quark[dil2] ;
  }
  qudaMemcpy(d_mom, host_mom, data_mom_bytes, qudaMemcpyHostToDevice);  
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_H2D);

  // doing too much work here as (di1,dil2) == (dil2,dil1)*
  int n_in_block = 0 , idx_last = 0 ;
  for (int dil1=0; dil1<n1; dil1++) {
    cpu_quark_bar_param.v = host_quark_bar[dil1] ;
    ColorSpinorField quark_bar(cpu_quark_bar_param) ;
    ColorSpinorField quda_quark_bar(cuda_quark_bar_param) ;
    quda_quark_bar = quark_bar ;
    // just block dil2
    for (int dil2=0; dil2<n2; dil2++){
      getProfileCurrentKernel().TPSTART(QUDA_PROFILE_COMPUTE);
      innerProductQuda( quda_quark_bar, quda_quark[dil2], (std::complex<double>*)d_tmp+n_sites*n_in_block );
      getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_COMPUTE);
      n_in_block++ ;
      if( n_in_block == block_size_mom_proj ) {
	getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
	blas_lapack::native::stridedBatchGEMM( d_mom, d_tmp,
					       (std::complex<double>*)d_ret+idx_last*X[3]*nMom,
					       cublas_param_mom_sum,
					       QUDA_CUDA_FIELD_LOCATION);
	idx_last += block_size_mom_proj ;
	n_in_block = 0 ;
	getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
      }
    }
  }
  // Copy device data back to host
  qudaMemcpy(return_array, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost) ;
  
  // Clean up memory allocations
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_ret);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_FREE);

  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_TOTAL);
}
