#include <color_spinor_field.h>
#include <kernels/color_contract.cuh>
#include <contract_quda.h>

#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor> class InnerProductV : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x;
    cvector_ref<const ColorSpinorField> &y;
    complex<Float> *result;
    unsigned int minThreads() const { return x.VolumeCB(); }    
  public:
    InnerProductV(const ColorSpinorField &x, cvector_ref<const ColorSpinorField> &y, void *result) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      result(static_cast<complex<Float>*>(result))
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorContractVArg<Float, nColor> arg(x, y, result);
      launch<InnerProdV>(tp, stream, arg);
    }
    
    long long flops() const
    {
      // 1 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 16 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + 16*y[0].Bytes() + 16 * x.Volume() * sizeof(Complex);
    }
  };

  void innerProductQudaV(const ColorSpinorField &x, cvector_ref<const ColorSpinorField> &y, void *result)
  {
    instantiate<InnerProductV>(x, y, result);
  }

  template <typename Float, int nColor> class ColorContractV : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x ;
    cvector_ref<const ColorSpinorField> &y;
    complex<Float> *result;
    unsigned int minThreads() const { return x.VolumeCB(); }
  public:
    ColorContractV( const ColorSpinorField &x, cvector_ref< const ColorSpinorField> &y, void *result ) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      result(static_cast<complex<Float>*>(result))
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorContractVArg<Float, nColor> arg(x, y, result);
      launch<ColorContractionV>(tp, stream, arg);
    }
    
    long long flops() const
    {
      return 16*3*6ll*x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + 16*y[0].Bytes() + 16*x.Volume() * sizeof(Complex);
    }
  };

  void colorContractQudaV(const ColorSpinorField &x, cvector_ref< const ColorSpinorField > &y, void *result)
  {
    instantiate<ColorContractV>(x, y, result);
  }

  // color cross mrhs on the y argument
  template <typename Float, int nColor> class ColorCrossV : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x;
    cvector_ref< const ColorSpinorField > &y;
    cvector_ref< ColorSpinorField > &result;
    size_t ny ;
    unsigned int minThreads() const { return x.VolumeCB(); }
  public:
    ColorCrossV(const ColorSpinorField &x, cvector_ref< const ColorSpinorField > &y, cvector_ref< ColorSpinorField > &result) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      ny(y.size()),
      result(result)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorCrossVArg<Float, nColor> arg(x, y, result);      
      launch<ColorCrossComputeV>(tp, stream, arg);
    }
    
    long long flops() const
    {
      return ny * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + ny*y[0].Bytes() + ny*x.Volume() * sizeof(complex<Float>);
    }
  };

  void colorCrossQudaV(const ColorSpinorField &x, cvector_ref< const ColorSpinorField > &y, cvector_ref< ColorSpinorField > &result)
  {
    checkPrecision(x, y); checkPrecision(result, y);
    if( y.size() != result.size() ) errorQuda( "Incompatible y and result sizes %zu != %zu\n" , y.size() , result.size() ) ; 
    instantiate<ColorCrossV>(x, y, result);
  }
}// namespace quda