#pragma once
#include "fvector.hh"

struct FuncStruct
{
  typedef float (*floatfunc_t)(const float& f);
  typedef fvector<4>(*fvect4func_t)(const fvector<4>& f);
  typedef fvector<8>(*fvect8func_t)(const fvector<8>& f);
  floatfunc_t func;
  floatfunc_t deriv;
  fvect4func_t func4;
  fvect4func_t deriv4;
  fvect8func_t func8;
  fvect8func_t deriv8;
  std::string name;
  
  template<typename T>
  using retfunc = T(*)(const T&);
};

template<typename T> // primary template
FuncStruct::retfunc<T> getFunc(const FuncStruct&);

template<>
inline
FuncStruct::retfunc<float> getFunc(const FuncStruct& fs)
{
  return fs.func;
}

template<>
inline
FuncStruct::retfunc<fvector<4>> getFunc(const FuncStruct& fs)
{
  return fs.func4;
}

template<> inline
FuncStruct::retfunc<fvector<8>> getFunc(const FuncStruct& fs)
{
  return fs.func8;
}

template<typename T> // primary template
FuncStruct::retfunc<T> getDeriv(const FuncStruct&);

template<> inline
FuncStruct::retfunc<float> getDeriv(const FuncStruct& fs)
{
  return fs.deriv;
}

template<> inline
FuncStruct::retfunc<fvector<4>> getDeriv(const FuncStruct& fs)
{
  return fs.deriv4;
}


template<> inline
FuncStruct::retfunc<fvector<8>> getDeriv(const FuncStruct& fs)
{
  return fs.deriv8;
}

inline FuncStruct MakeExpFunc()
{
  FuncStruct ret;
  ret.func = [](const float& f) -> float { return expf(f);};
  ret.deriv = [](const float& f) -> float { return expf(f);};
  ret.func4 = [](const fvector<4>& f) { return exp(f);};
  ret.deriv4 = [](const fvector<4>& f) -> fvector<4> { return exp(f);};
  ret.func8 = [](const fvector<8>& f) { return exp(f);};
  ret.deriv8 = [](const fvector<8>& f) -> fvector<8> { return exp(f);};
  ret.name="exp";
  return ret;
}

inline FuncStruct MakeSquareFunc()
{
  FuncStruct ret;
  ret.func = [](const float& f) -> float { return f*f;};
  ret.deriv = [](const float& f) -> float { return 2*f;};
  ret.func4 = [](const fvector<4>& f) { return f*f;};
  ret.deriv4 = [](const fvector<4>& f) -> fvector<4> { return 2*f;};
  ret.func8 = [](const fvector<8>& f) { return f*f;};
  ret.deriv8 = [](const fvector<8>& f) -> fvector<8> { return 2*f;};
  ret.name="square";
  return ret;
}

inline FuncStruct MakeReluFunc()
{
  FuncStruct ret;
  ret.func = [](const float& f) -> float { return std::max(0.0F, f); };
  ret.deriv = [](const float& f) -> float { return f < 0.0 ? 0.0F : 1.0F; };
  ret.func4 = [](const fvector<4>& in) {
    fvector<4> cmp = 0;
    return (cmp < in) * in;
  };
  ret.deriv4 = [](const fvector<4>& in) {
    fvector<4> cmp = 0;
    return (cmp < in) * 1.0;
  };
  ret.func8 = [](const fvector<8>& in) {
    fvector<8> cmp = 0;
    return (cmp < in) * in;
  };
  ret.deriv8 = [](const fvector<8>& in) {
    fvector<8> cmp = 0;
    return (cmp < in) * 1.0;
  };
  ret.name="relu";
  return ret;
}


inline FuncStruct MakeTanhFunc()
{
  FuncStruct ret;
  ret.func = [](const float& in) { return tanhf(in);  };
  ret.deriv = [](const float& in)  {  float t = tanh(in); return 1-t*t;  };

  ret.func8 = [](const fvector<8>& in) { return tanh(in); };
  ret.deriv8 =[](const fvector<8>& in) { fvector<8> t = tanh(in); return fvector<8>(1.0) - t*t;  };
  ret.name = "tanh";
  return ret;
};


inline FuncStruct MakeSigmoidFunc()
{
  FuncStruct ret;
  ret.func8 = [](const fvector<8>& in)
  {
    return 1.0F / (fvector<8>(1.0F) + exp(-in));
  };
  ret.deriv8 = [](const fvector<8>& in)
  {
    fvector<8> sigma = 1.0F / (fvector<8>(1.0F) + exp(-in));
    return sigma * (fvector<8>(1.0F) - sigma);
  };

  ret.func = [](const float& in)
  {
    return 1.0F / (1.0F + expf(-in));
  };
  ret.deriv = [](const float& in)
  {
    float sigma = 1.0F / (1.0F + expf(-in));
    return sigma * (1.0F - sigma);
  };
  ret.name="sigmoid";
  return ret;
};


inline FuncStruct MakeLogFunc()
{
  struct FuncStruct ret;
  
  ret.func = [](const float& in)
  {
    if (in == 0.0F)
      return -80.0F;
    return logf(in);
  };
  ret.deriv = [](const float& in)
  {
    if (in == 0.0F)
      return 80.0F;
    return 1.0F / in;
  };

  ret.func8 = [](const fvector<8>& in)
  {
    return log(in);
  };
  ret.deriv8 = [](const fvector<8>& in)
  {
    return 1.0/in;
  };
  
  ret.name = "log";
  return ret;
};

static std::vector<FuncStruct> g_fss({
    MakeSigmoidFunc(), // 0
    MakeReluFunc(),    // 1
    MakeExpFunc(),     // 2 
    MakeLogFunc(),     // 3
    MakeTanhFunc(),    // 4
    MakeSquareFunc()});// 5

inline uint8_t SigmoidFunc(){ return 0; }
inline uint8_t ReluFunc(){ return 1; }
inline uint8_t ExpFunc(){ return 2; }
inline uint8_t LogFunc(){ return 3; }
inline uint8_t TanhFunc(){ return 4; }
inline uint8_t SquareFunc(){ return 5; }


