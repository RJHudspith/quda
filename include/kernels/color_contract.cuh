#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>

#include <constant_kernel_arg.h>
#include <kernel.h>

namespace quda {

  // right hand side batch sizes I have tuned
  constexpr unsigned long max_color_n = 16 , max_cross_n = 16;

  // MRHS version
  template <typename Float, int nColor_> struct ColorCrossVArg : kernel_param<> {
    using real = typename mapper<Float>::type ;
    static constexpr int nColor = nColor_ , nSpin = 1;
    typedef typename colorspinor_mapper<Float, 1, nColor_, false, false>::type F;
    F x , y[ max_cross_n ] , result[ max_cross_n ] ;  
    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    int_fastdiv ny = (int_fastdiv)max_cross_n ;
    ColorCrossVArg(const ColorSpinorField &x, cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &result) :
      x(x),
      ny(y.size()),
      threads(x.VolumeCB())
    {
    if( y.size() != result.size() ) errorQuda( "y and result must be same size %zu != %zu" , y.size() , result.size() ) ;
    	if( y.size() > max_cross_n || result.size() > max_cross_n ) errorQuda( "Max vector length in colorCrossV is %d" , max_cross_n ) ;
      for(int i=0; i<4; i++) X[i] = x.X()[i];
      for(auto i=0u; i< y.size() ; i++) this -> y[i] = y[i] ;
      for(auto i=0u; i< result.size() ; i++) this -> result[i] = result[i] ;
    }
  };
    
  template <typename Float, int nColor_> struct ColorContractVArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_ , nSpin = 1;
    // Create a typename F for the ColorSpinorFields
    typedef typename colorspinor_mapper<Float, nSpin, nColor, false, false>::type F;
    F x, y[ max_color_n ] ;
    complex<Float> *s;
    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    int_fastdiv ny = 1 , V = 1 ;
    ColorContractVArg(const ColorSpinorField &x, cvector_ref<const ColorSpinorField> &y, complex<Float> *s) :
      x(x),
      ny(y.size()),
      s(s),
      threads(x.VolumeCB()),
      V(2*x.VolumeCB())
    {
      if( y.size() > max_color_n ) errorQuda( "Too large batching factor in y for InnerProductV %zu" , y.size() ) ;
      for(int i=0; i<4; i++) { X[i] = x.X()[i]; }      
      for(size_t i = 0 ; i < y.size() ; i++ ) this -> y[i] = y[i] ;
    }
  };

  template <typename Arg> struct InnerProdV {
    const Arg &arg;
    constexpr InnerProdV(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;
      Vector x = arg.x(x_cb, parity) ;
      const size_t V = 2*arg.threads.x ;
      for( int i = 0 ; i < arg.ny ; i++ ) {
         Vector y = arg.y[i](x_cb, parity);
         arg.s[x_cb + parity*(arg.threads.x)+i*V] = innerProduct(x, y, 0, 0) ;
      }
    }
  };

  template <typename Arg> struct ColorContractionV {
    const Arg &arg;
    constexpr ColorContractionV(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;
      Vector x = arg.x(x_cb, parity) ;
      const size_t V = 2*arg.threads.x ;
      for( int i = 0 ; i < arg.ny ; i++ ) {
         Vector y = arg.y[i](x_cb, parity);
         arg.s[x_cb + parity*(arg.threads.x)+i*V] = colorContract(x, y, 0, 0) ;
      }
    }
  };

  template <typename Arg> struct ColorCrossComputeV {
    const Arg &arg;
    constexpr ColorCrossComputeV(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;
      Vector x = arg.x(x_cb, parity);
      #pragma unroll
      for( auto i = 0 ; i < arg.ny ; i++ ) {
          Vector y = arg.y[i](x_cb, parity);
      	  Vector result = crossProduct(x, y, 0, 0);
	  arg.result[i](x_cb, parity) = result;
      }
    }
  };
}