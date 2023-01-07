#pragma once
#include <array>
#include <numeric>
#include <iostream>

template<unsigned int W>
struct fvector 
{
  float v __attribute__((vector_size (4*W)));
  fvector()
  {}
  fvector(float v0)
  {
    for(unsigned int n=0 ; n < W; ++n)
      v[n] = v0; 
  }

  // be VERY careful, fvector<8>{0.0} will call this
  fvector(const std::initializer_list<float>& in)
  {
    unsigned char ctr=0;
    for(const auto& val: in)
      v[ctr++] = val;
    for(; ctr < W; ++ctr)
      v[ctr] = 0.0;
  }

  fvector& operator=(float val)
  {
    *this = fvector(val);
    return *this;
  }

  fvector operator+(const fvector& rhs)
  {
    fvector ret = *this;
    ret.v += rhs.v;
    return ret;
  }
  fvector operator-(const fvector& rhs) const
  {
    fvector ret = *this;
    ret.v -= rhs.v;
    return ret;
  }
  fvector operator-() const
  {
    fvector ret = *this;
    ret.v = -ret.v;
    return ret;
  }

  fvector operator!() const
  {
    fvector ret = *this;
    for(unsigned int n = 0 ; n < W; ++n)
      ret.v[n] = !v[n];

    return ret;
  }

  fvector operator*(const float& rhs) const
  {
    fvector ret = *this;
    ret.v *= rhs;
    return ret;
  }
  
  fvector operator*(const fvector& rhs) const
  {
    fvector ret = *this;
    ret.v *= rhs.v;
    return ret;
  }
  fvector operator/(const fvector& rhs) const
  {
    fvector ret = *this;
    ret.v /= rhs.v;
    return ret;
  }
  fvector& operator/=(const fvector& rhs)
  {
    v /= rhs.v;
    return *this;
  }
  fvector& operator*=(float rhs)
  {
    v *= rhs;
    return *this;
  }

  fvector& operator*=(const fvector& rhs)
  {
    v *= rhs.v;
    return *this;
  }
  fvector& operator+=(const fvector& rhs)
  {
    v += rhs.v;
    return *this;
  }
  fvector& operator+=(float rhs)
  {
    v += rhs;
    return *this;
  }
  
  fvector operator<(const fvector& rhs) const
  {
    fvector ret;
    // you'd think you'd be able to do this:
    //    ret.v = (v < rhs.v);
    // but the output is Weird - if the comparison is valid, the output is 0 (!)
    // and if invalid, it is NaN (!!) https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html

    for(unsigned int i = 0 ; i < W; ++i)
      ret.v[i] = v[i] < rhs.v[i];

    return ret;
  }

  bool operator==(const fvector& rhs) const
  {
    for(unsigned int n=0; n < W; ++n)
      if(v[n] != rhs.v[n])
        return false;
    return true;
  }

  float sum() const
  {
    float ret = 0;
    for(unsigned int n=0; n < W; ++n)
      ret += v[n];
    return ret;
  }
};

template<unsigned int T>
auto operator*(float v, const fvector<T>& rhs)
{
  return rhs * v;
}

template<unsigned int T>
auto operator/(float v, const fvector<T>& rhs)
{
  fvector<T> ret = v;
  return ret/rhs;
}

template<unsigned int T>
auto operator+(float v, const fvector<T>& rhs)
{
  fvector<T> ret = v;
  return ret+rhs;
}


template<unsigned int W>
auto exp(const fvector<W>& v)
{
  fvector<W> ret;
  for(size_t i = 0; i < W ; ++i) {
    ret.v[i] = expf(v.v[i]);
  }
  return ret;
}

template<unsigned int W>
auto log(const fvector<W>& v)
{
  fvector<W> ret;
  for(size_t i = 0; i < W ; ++i) {
    ret.v[i] = logf(v.v[i]);
  }
  return ret;
}

template<unsigned int W>
auto tanh(const fvector<W>& v)
{
  fvector<W> ret;
  for(size_t i = 0; i < W ; ++i) {
    ret.v[i] = tanhf(v.v[i]);
  }
  return ret;
}


template<unsigned int W>
std::ostream& operator<<(std::ostream& os, const fvector<W>& o) 
{
  for(unsigned int n=0 ; n < W; ++n)
    os << o.v[n] <<" ";
  return os;
}
/*
template<unsigned int W>
void setZero(fvector<W>& o)
{
  o=0.0;
}
*/

template<unsigned int W>
auto maxFunc(const fvector<W>& lhs, const fvector<W>& rhs)
{
  fvector<W> ret;
  for(unsigned int i = 0 ; i < W; ++i)
    ret.v[i] = std::max(lhs.v[i], rhs.v[i]);
  return ret;
}

