#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <blas_quda.h>
#include <tune_quda.h>

#include <color_spinor_field.h>
#include <contract_quda.h>

#include <cassert>

using namespace quda;

static TimeProfile profileNKernel("NKernelQuda");
TimeProfile &getProfileNKernel() { return profileNKernel; }
static TimeProfile profileModeNlet("ModeNlet");
TimeProfile &getProfileModeNlet() { return profileModeNlet ; }
static TimeProfile profileApplyNoise("applyNoise");
TimeProfile &getProfileApplyNoise() { return profileApplyNoise; }
static TimeProfile profileColorContract("colorContractQuda");
TimeProfile &getProfileColorContract() { return profileColorContract; }
static TimeProfile profileColorCross("colorCrossQuda");
TimeProfile &getProfileColorCross() { return profileColorCross; } 
static TimeProfile profileInnerProduct("innerProduct");
TimeProfile &getProfileInnerProduct() { return profileInnerProduct; } 
TimeProfile &getProfileBLAS();

static const double OneGB = 1024.*1024.*1024.;

// blocking factors for color contractions and color cross
static const size_t nRHS = 16 , nDiq = 16 ;

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

// return d_ret to the host handling different precisions
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
	      std::vector<std::vector<ColorSpinorField>> &q ,
	      const std::vector<std::vector<std::complex<double>>> &coeffs )
{
  const size_t nEv = evec.size() ;
  std::vector<ColorSpinorField> quda_evec(1) ;
  quda_evec[0] = ColorSpinorField(cuda_evec_param);
  for (size_t i=0; i<nEv; i++) {
    quda_evec[0] = evec[i] ;
    #pragma unroll
    for( size_t n = 0 ; n < q.size() ; n++ ) {
      const size_t n1 = q[n].size() ;
      blas::block::caxpy( {coeffs[n].begin()+n1*i,coeffs[n].begin()+n1*(i+1)},
			  {quda_evec[0]}, {q[n].begin(),q[n].end()} ) ;
    }
  }
}

// because we now have the same indexing pattern all our BLAS DFT calls are the same
static QudaBLASParam
default_BLAS( const size_t nMom , const int X[4] , const int blockSizeMomProj , const int precision , const bool bDag = false )
{
  const int nSp = X[0]*X[1]*X[2] ;
  const int nSites = nSp*X[3] ;
  QudaBLASParam cublas_param = newQudaBLASParam() ;
  cublas_param.trans_a = QUDA_BLAS_OP_N;
  cublas_param.trans_b = bDag ? QUDA_BLAS_OP_C : QUDA_BLAS_OP_T;
  cublas_param.m = (int)nMom ;
  cublas_param.n = X[3] ;
  cublas_param.k = nSp/2 ;
  cublas_param.lda = nSp ;
  cublas_param.ldb = nSp ;
  cublas_param.ldc = X[3] ;
  cublas_param.a_stride = 0 ;
  cublas_param.b_stride = nSites ;
  cublas_param.c_stride = X[3]*(int)nMom ;
  cublas_param.batch_count = blockSizeMomProj;
  cublas_param.alpha = 1. ; cublas_param.beta = 0. ;
  cublas_param.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param.data_type = ( precision == QUDA_SINGLE_PRECISION ) ? \
    QUDA_BLAS_DATATYPE_C : QUDA_BLAS_DATATYPE_Z;
  cublas_param.blas_type = QUDA_BLAS_GEMM ;
  return cublas_param ;
}

// compute the DFT
static inline void
doBlasReturn( QudaBLASParam cublas_dft ,
	      void *d_tmp , void *d_ret , void *d_mom , 
	      double _Complex *return_array ,
	      size_t &nInBlock , size_t &blockStart ,
	      const size_t nMom , const int X[4] , const int precision )
{
  const int nSp = X[0]*X[1]*X[2] ;
  cublas_dft.batch_count = (int)nInBlock;
  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);  
  blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, d_ret,
					cublas_dft, QUDA_CUDA_FIELD_LOCATION);
  cublas_dft.beta = 1. ;
  blas_lapack::native::stridedBatchGEMM( (char*)d_mom + nSp*precision,
					 (char*)d_tmp + nSp*precision,
					 d_ret, cublas_dft,
					 QUDA_CUDA_FIELD_LOCATION);
  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
  hostreturn( d_ret , return_array + X[3]*nMom*blockStart ,
	      (size_t)nInBlock*X[3]*nMom , precision ) ;
  blockStart += nInBlock; nInBlock = 0;
}

// allocate DFT device arrays
static void
allocateMomDevice( void **d_tmp , void **d_ret , void **d_mom ,
		   const size_t nRHS , const size_t blockSizeMomProj ,
		   const int X[4] , const int precision , const size_t nMom )
{
  const size_t nSp = X[0]*X[1]*X[2] ;
  const size_t nSites = nSp*X[3] ;
  const size_t data_tmp_bytes = std::max( nRHS , blockSizeMomProj )*nSites*2*precision;
  const size_t data_ret_bytes = nMom*blockSizeMomProj*X[3]*2*precision;
  const size_t data_mom_bytes = nMom*nSp*2*precision;
  *d_tmp = pool_device_malloc(data_tmp_bytes);
  *d_ret = pool_device_malloc(data_ret_bytes);
  *d_mom = pool_device_malloc(data_mom_bytes);
  if( getVerbosity() >= QUDA_SUMMARIZE ) {
    const size_t OneGB = 1024*1024*1024;
    const size_t total_bytes = data_tmp_bytes + data_ret_bytes + data_mom_bytes ;
    printfQuda("d_tmp %fGB | d_ret %fGB | d_mom %fGB | total = %fGB\n",
	       (double)data_tmp_bytes/(OneGB), (double)data_ret_bytes/(OneGB),
	       (double)data_mom_bytes/(OneGB), (double)total_bytes/(OneGB)); 
  }
}

// free mom device memory
static void
freeMomDevice( void **d_tmp , void **d_ret , void **d_mom )
{
  pool_device_free( *d_tmp ) ;
  pool_device_free( *d_ret ) ;
  pool_device_free( *d_mom ) ;
}

// set the transposed noise values on the device
static void
setQudaQ( const size_t *nDil ,
	  const double _Complex **host_coeffs ,
	  const size_t nEv ,
	  ColorSpinorParam cuda_evec_param ,
	  QudaInvertParam inv_param ,
	  std::vector<ColorSpinorParam> &quda_q_param ,
	  std::vector<std::vector<std::complex<double>>> &coeffs ,
	  std::vector<std::vector<ColorSpinorField>> &quda_q )
{
  const size_t N = coeffs.size() ;
  for( size_t d = 0 ; d < N ; d++ ) {
    quda_q_param[d] = ColorSpinorParam( cuda_evec_param, inv_param, QUDA_CUDA_FIELD_LOCATION );
    quda_q_param[d].create = QUDA_ZERO_FIELD_CREATE ;
    quda_q[d].resize( nDil[d] ) ; coeffs[d].resize( nDil[d]*nEv) ;
    for(size_t i = 0 ; i < nDil[d] ; i++) {
      quda_q[d][i] = ColorSpinorField( quda_q_param[d] ) ;
      for( int j = 0 ; j < nEv ; j++ ) { coeffs[d][i+j*nDil[d]] = (std::complex<double>)host_coeffs[d][j+i*nEv] ; }
    }
  }
}

void nKernel( const size_t *nDil ,
	      const double _Complex **host_coeffs,
	      const size_t nMom,
	      const double _Complex *host_mom, 
	      const size_t nEv,
	      void **host_evec,
	      QudaInvertParam inv_param,
	      double _Complex *return_array,
	      const size_t blockSizeMomProj,
	      const int X[4] ,
	      const int N )
{
  getProfileNKernel().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileNKernel().TPSTART(QUDA_PROFILE_INIT);
  assert( N < 5 && N > 1 ) ;
  size_t nProd = 1 ;
  for( size_t i  = 0 ; i < N ; i++ ) {
    nProd *= nDil[i] ;
  }
  if( sizeof(Complex) != sizeof(double _Complex) ) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( blockSizeMomProj > nProd ) {
    errorQuda( "Block size mom proj %zu > %zu\n", blockSizeMomProj, nProd ) ;
  }
  if( inv_param.cuda_prec != QUDA_DOUBLE_PRECISION &&
      inv_param.cuda_prec != QUDA_SINGLE_PRECISION ) {
    errorQuda( "Unsupported device precision %d" , inv_param.cuda_prec ) ;
  }
  const size_t nSp    = X[0]*X[1]*X[2] ;
  const size_t nSites = nSp*X[3] ;
  const int precision = inv_param.cuda_prec ;
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
  std::vector<ColorSpinorParam> quda_q_param( N ) ;
  std::vector<std::vector<std::complex<double>>> coeffs( N ) ;
  std::vector<std::vector<ColorSpinorField>> quda_q( N ) ;
  setQudaQ( nDil , host_coeffs , nEv , cuda_evec_param , inv_param ,
	    quda_q_param , coeffs , quda_q ) ;
  void *d_tmp = NULL , *d_ret = NULL , *d_mom = NULL ; 
  allocateMomDevice( &d_tmp , &d_ret , &d_mom , nDiq , blockSizeMomProj , X , precision , nMom ) ;
  getProfileNKernel().TPSTOP(QUDA_PROFILE_INIT);
  // Copy host_mom data to device
  getProfileNKernel().TPSTART(QUDA_PROFILE_H2D);
  device_hostmom( host_mom , d_mom , nMom*nSp , precision ) ;
  getProfileNKernel().TPSTOP(QUDA_PROFILE_H2D);
  // apply the noises to the evecs
  getProfileApplyNoise().TPSTART(QUDA_PROFILE_COMPUTE);
  apply_noises( evec , cuda_evec_param , quda_q , coeffs ) ;
  getProfileApplyNoise().TPSTOP(QUDA_PROFILE_COMPUTE);
  // usual momentum contraction, strided and blocked
  ColorSpinorParam cuda_diq_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  size_t nInBlock = 0, blockStart = 0;
  getProfileNKernel().TPSTART(QUDA_PROFILE_COMPUTE);
  switch( N ) {
  case 4 : {
    QudaBLASParam cublas_dft = default_BLAS( nMom , X , blockSizeMomProj , precision , true ) ;
    std::vector< ColorSpinorField > quda_diq1( 1 ) , quda_diq2( nDiq ) ;
    quda_diq1[0] = ColorSpinorField( cuda_diq_param ) ;
    for( size_t i = 0 ; i < quda_diq2.size() ; i++ ) {
      quda_diq2[i] = ColorSpinorField( cuda_diq_param ) ;
    }
    for( size_t aEv=0; aEv<nDil[0] ; aEv++) {
      for( size_t bEv=0; bEv<nDil[1]; bEv++ ) {
	const size_t diqBlk1 = 1 ;
	getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
	colorCrossQudaV( quda_q[0][aEv],
			 { quda_q[1].begin() + bEv , quda_q[1].begin() + bEv + diqBlk1 } ,
			 { quda_diq1.begin() , quda_diq1.begin()+diqBlk1 } ) ;
	getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
	for(size_t cEv=0; cEv<nDil[2]; cEv++ ) {
	  size_t dEv = 0 ;
	  while( dEv < nDil[3] ) {
	    const int diqBlk2 = std::min( std::min( nDil[3] - dEv , nDiq ) , (blockSizeMomProj-nInBlock ) ) ; ;
	    getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
	    colorCrossQudaV( quda_q[2][cEv],
			     { quda_q[3].begin() + dEv , quda_q[3].begin() + dEv + diqBlk2 } ,
			     { quda_diq2.begin() , quda_diq2.begin()+diqBlk2 } ) ;      
	    getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
	    getProfileInnerProduct().TPSTART(QUDA_PROFILE_COMPUTE);
	    innerProductQudaV( quda_diq1[0], { quda_diq2.begin() , quda_diq2.begin()+diqBlk2 } ,
			       (char*)d_tmp+nSites*nInBlock*2*precision );
	    getProfileInnerProduct().TPSTOP(QUDA_PROFILE_COMPUTE);
	    nInBlock += diqBlk2 ; dEv += diqBlk2 ;
	    if (nInBlock == blockSizeMomProj) {
	      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
			    nInBlock , blockStart , nMom , X , precision ) ;
	    }
	  }
	}
      }
    }
    if( nInBlock > 0 ) {
      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
		    nInBlock , blockStart , nMom , X , precision ) ;
    }
  } break ;
  case 3 : {
    QudaBLASParam cublas_dft = default_BLAS( nMom , X , blockSizeMomProj , precision , false ) ;
    std::vector<ColorSpinorField> quda_diq( nDiq ) ;
    for( size_t i = 0 ; i < quda_diq.size() ; i++ ) {
      quda_diq[i] = ColorSpinorField( cuda_diq_param ) ;
    }
    for( size_t dil1=0; dil1<nDil[0]; dil1++ ) {
      size_t dil2 = 0 ;
      while( dil2 < nDil[1] ) {
	const size_t diqblk = std::min( nDiq , nDil[1]-dil2 ) ;
	getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
	colorCrossQudaV( quda_q[0][dil1], { quda_q[1].begin() + dil2 , quda_q[1].begin() + dil2 + diqblk } ,
			 { quda_diq.begin() , quda_diq.begin()+diqblk } );
	getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
	for( size_t b = 0 ; b < diqblk ; b++ ) {
	  size_t dil3 = 0 ;
	  while( dil3 < nDil[2] ) {
	    const size_t blk = std::min( std::min( nDil[2] - dil3 , nRHS ) , (blockSizeMomProj-nInBlock ) ) ;
	    getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);
	    colorContractQudaV( quda_diq[b], { quda_q[2].begin() + dil3 , quda_q[2].begin() + dil3 + blk } ,
				(char*)d_tmp + nSites*nInBlock*2*precision);
	    getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	    nInBlock+=blk ; dil3 += blk ;
	    if (nInBlock == blockSizeMomProj) {
	      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
			    nInBlock , blockStart , nMom , X , precision ) ;
	    }
	  }
	}
	dil2 += diqblk ;
      }
    }
    if( nInBlock > 0 ) {
      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
		    nInBlock , blockStart , nMom , X , precision ) ;
    }
  } break ;
  case 2 : {
    QudaBLASParam cublas_dft = default_BLAS( nMom , X , blockSizeMomProj , precision , false ) ;
    for( size_t dil1=0; dil1<nDil[0]; dil1++) {
      size_t dil2 = 0 ;
      while( dil2 < nDil[1] ) {
	const size_t blk = std::min( std::min( nDil[1] - dil2 , nRHS ) , blockSizeMomProj-nInBlock ) ;
	getProfileInnerProduct().TPSTART(QUDA_PROFILE_COMPUTE);
	innerProductQudaV( quda_q[0][dil1], { quda_q[1].begin() + dil2 , quda_q[1].begin() + dil2 + blk },
			   (char*)d_tmp+nSites*nInBlock*2*precision );
	getProfileInnerProduct().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock+=blk ; dil2 += blk ;
	if( nInBlock == blockSizeMomProj ) {
	  doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
			nInBlock , blockStart , nMom , X , precision ) ;
	}
      }
    }
    if( nInBlock > 0 ) {
      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
		    nInBlock , blockStart , nMom , X , precision ) ;
    }
  } break ;
  default : errorQuda( "Should not get here" ) ; break ;
  }
  getProfileNKernel().TPSTOP(QUDA_PROFILE_COMPUTE);
  // Clean up memory allocations
  getProfileNKernel().TPSTART(QUDA_PROFILE_FREE);
  freeMomDevice( &d_tmp , &d_ret , &d_mom ) ;
  getProfileNKernel().TPSTOP(QUDA_PROFILE_FREE);
  getProfileNKernel().TPSTOP(QUDA_PROFILE_TOTAL);
}

// mode doublet version
void modeNlet( const size_t nMom,
	       const double _Complex *host_mom,
	       const size_t nEv,
	       void **host_evec,
	       QudaInvertParam inv_param,
	       double _Complex *return_array,
	       const size_t blockSizeMomProj,
	       const int X[4],
	       const int N )
{
  getProfileModeNlet().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileModeNlet().TPSTART(QUDA_PROFILE_INIT);
  size_t nEvC =	1 , den = 1 ;
  for( int i = N-1 ; i >= 0 ; i-- ) {
    nEvC *= (nEv-i) ; den *= (i+1) ; 
  }
  nEvC /= den ;
  if( blockSizeMomProj > nEvC ) {
    errorQuda("block_size_mom_proj %zu > C(nEv,%d) -> %d" , blockSizeMomProj, N , nEvC ) ;
  }
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  if( inv_param.cuda_prec != QUDA_DOUBLE_PRECISION &&
      inv_param.cuda_prec != QUDA_SINGLE_PRECISION ) {
    errorQuda("Unsupported device precision") ;
  }
  const QudaPrecision precision = inv_param.cuda_prec ;
  const size_t nSp    = X[0]*X[1]*X[2];
  const size_t nSites = nSp*X[3];
  const lat_dim_t x   = { X[0] , X[1] , X[2] , X[3] } ;
  ColorSpinorParam cpu_evec_param(host_evec, inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_evec_param.nSpin = 1;
  std::vector<ColorSpinorField> cpu_evec(nEv) ;
  for( int ev = 0 ; ev < nEv ; ev++ ) {
    cpu_evec_param.v = host_evec[ev] ;
    cpu_evec[ev] = ColorSpinorField(cpu_evec_param) ;
  }
  ColorSpinorParam cuda_quark_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  cuda_quark_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  void *d_tmp = NULL , *d_ret = NULL , *d_mom = NULL ; 
  allocateMomDevice( &d_tmp , &d_ret , &d_mom , nRHS , blockSizeMomProj , X , precision , nMom ) ;
  getProfileModeNlet().TPSTOP(QUDA_PROFILE_INIT);
  // Copy host data to device for all evecs
  getProfileModeNlet().TPSTART(QUDA_PROFILE_H2D);
  std::vector<ColorSpinorField> quda_evec(nEv) ;
  for (int ev=0; ev<nEv; ev++) {
    quda_evec[ev] = ColorSpinorField(cuda_quark_param) ;
    quda_evec[ev] = cpu_evec[ev] ;
  }
  device_hostmom( host_mom , d_mom , nMom*nSp , precision ) ;
  getProfileModeNlet().TPSTOP(QUDA_PROFILE_H2D);
  size_t nInBlock = 0 , blockStart = 0 ;
  ColorSpinorParam cuda_diq_param(cpu_evec_param,inv_param,QUDA_CUDA_FIELD_LOCATION);
  getProfileModeNlet().TPSTART(QUDA_PROFILE_COMPUTE);
  switch( N ) {
    // mode quartet
    // generic 4-quark object \phi^{i,j,k,l} = \epsilon_{abc}\epsilon_{ade} v^i_b v^j_c (v^k_d)^* (v^l_e)^*
  case 4 : {
    QudaBLASParam cublas_dft = default_BLAS( nMom , X , blockSizeMomProj , precision , true ) ;
    std::vector< ColorSpinorField > quda_diq1( 1 ) , quda_diq2( nDiq ) ;
    quda_diq1[0] = ColorSpinorField( cuda_diq_param ) ;
    for( size_t i = 0 ; i < quda_diq2.size() ; i++ ) {
      quda_diq2[i] = ColorSpinorField( cuda_diq_param ) ;
    }
    for( size_t aEv=0; aEv<nEv; aEv++) {
      for( size_t bEv=aEv+1; bEv<nEv; bEv++ ) {
	const size_t diqBlk1 = 1 ;
	getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
	colorCrossQudaV( quda_evec[aEv],
			 { quda_evec.begin() + bEv , quda_evec.begin() + bEv + diqBlk1 } ,
			 { quda_diq1.begin() , quda_diq1.begin()+diqBlk1 } ) ;      
	getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
	for( size_t cEv=bEv+1; cEv<nEv; cEv++ ) {
	  size_t dEv = cEv+1 ;
	  while( dEv < nEv ) {
	    // blocking factor for multi RHS colorCross and inner product
	    const size_t diqBlk2 = std::min( std::min( nEv - dEv , nDiq ) , (blockSizeMomProj-nInBlock ) ) ; ;
	    getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
	    colorCrossQudaV( quda_evec[cEv],
			     { quda_evec.begin() + dEv , quda_evec.begin() + dEv + diqBlk2 } ,
			     { quda_diq2.begin() , quda_diq2.begin()+diqBlk2 } ) ;      
	    getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
	    getProfileInnerProduct().TPSTART(QUDA_PROFILE_COMPUTE);
	    innerProductQudaV( quda_diq1[0], { quda_diq2.begin() , quda_diq2.begin()+diqBlk2 } ,
			       (char*)d_tmp+nSites*nInBlock*2*precision ) ;
	    getProfileInnerProduct().TPSTOP(QUDA_PROFILE_COMPUTE);
	    nInBlock+=diqBlk2 ; dEv += diqBlk2 ;
	    if (nInBlock == blockSizeMomProj) {
	      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
			    nInBlock , blockStart , nMom , X , precision ) ;
	    }
	  }
	}
      }
    }
    if( nInBlock > 0 ) {
      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
		    nInBlock , blockStart , nMom , X , precision ) ;
    }
  } break ;
    // mode triplets
    // generic 3-quark object \phi^{i,j,k} = \epsilon_{abc} v^i_a v^j_b v^k_c
  case 3 : {
    QudaBLASParam cublas_dft = default_BLAS( nMom , X , blockSizeMomProj , precision , false ) ;
    std::vector<ColorSpinorField> quda_diq(nDiq) ;
    for( size_t i = 0 ; i < quda_diq.size() ; i++ ) {
      quda_diq[i] = ColorSpinorField( cuda_diq_param ) ;
    }
    for (int aEv=0; aEv<nEv; aEv++) {
      size_t bEv = aEv+1 ;
      while( bEv < nEv ) {
	const int diqBlk = std::min( nEv - bEv , nDiq ) ;
	getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
	colorCrossQudaV( quda_evec[aEv],
			 { quda_evec.begin() + bEv , quda_evec.begin() + bEv + diqBlk } ,
			 { quda_diq.begin() , quda_diq.begin()+diqBlk } ) ;
	getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
	for( int b = 0 ; b < diqBlk ; b++ ) {
	  int cEv = bEv+1+b ;
	  while( cEv < nEv ) {
	    const int blk = std::min( std::min( nEv - cEv , nRHS ) , (blockSizeMomProj-nInBlock ) ) ;
	    getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);
	    colorContractQudaV( quda_diq[b], { quda_evec.begin() + cEv , quda_evec.begin() + cEv + blk } ,
				(char*)d_tmp + nSites*nInBlock*2*precision);
	    getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	    nInBlock+=blk ; cEv += blk ;
	    if (nInBlock == blockSizeMomProj) {
	      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
			    nInBlock , blockStart , nMom , X , precision ) ;
	    }
	  }
	}// diqBlk
	bEv += diqBlk ;
      }
    }
    if( nInBlock > 0 ) {
      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
		    nInBlock , blockStart , nMom , X , precision ) ;
    }
  } break ;
    // mode doublet i.e. a meson or something
    // generic 2-quark object \phi^{i,j} = v^i_a (v^j_a)^*
  case 2 : {
    QudaBLASParam cublas_dft = default_BLAS( nMom , X , blockSizeMomProj , precision , false ) ;
    for( size_t dil1=0; dil1<nEv; dil1++) {
      size_t dil2 = 0 ;
      while( dil2 < nEv ) {
	const int blk = std::min( std::min( nEv-dil2 , nRHS ) , blockSizeMomProj-nInBlock ) ;
	getProfileInnerProduct().TPSTART(QUDA_PROFILE_COMPUTE);
	innerProductQudaV( quda_evec[dil1],
			   { quda_evec.begin() + dil2, quda_evec.begin() + dil2 + blk } ,
			   (char*)d_tmp+nSites*nInBlock*2*precision );
	getProfileInnerProduct().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock += blk ; dil2 += blk ;
	if( nInBlock == blockSizeMomProj ) {
	  doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
			nInBlock , blockStart , nMom , X , precision ) ;
	}
      }
    }
    if( nInBlock > 0 ) {
      doBlasReturn( cublas_dft , d_tmp , d_ret , d_mom , return_array ,
		    nInBlock , blockStart , nMom , X , precision ) ;
    }
  } break ;
  default :
    errorQuda( "Should not get here\n" ) ;
    break ;
  }
  getProfileModeNlet().TPSTART(QUDA_PROFILE_COMPUTE);
  // Clean up memory allocations
  getProfileModeNlet().TPSTART(QUDA_PROFILE_FREE);
  freeMomDevice( &d_tmp , &d_ret , &d_mom ) ;
  getProfileModeNlet().TPSTOP(QUDA_PROFILE_FREE);
  getProfileModeNlet().TPSTOP(QUDA_PROFILE_TOTAL);
}
