#pragma once
#include <array>
#include <numeric>

template<unsigned int W>
union fvector 
{
  typedef float otype __attribute__((vector_size (4*W)));
  otype v;
  std::array<float, W> a;
  fvector()
  {}
  fvector(float v0)
  {
    *this = v0;
  }
  fvector(const std::initializer_list<float>& in)
  {
    unsigned int ctr=0;
    for(const auto& val: in)
      a[ctr++] = val;
  }

  fvector& operator=(float val)
  {
    for(auto& i : a)
      i = val;
    return *this;
  }
  float sum() const
  {
    return std::accumulate(a.cbegin(), a.cend(), 0.0);
  }

  fvector operator+(const fvector& rhs)
  {
    fvector ret = *this;
    ret.v += rhs.v;
    return ret;
  }
  fvector operator-(const fvector& rhs)
  {
    fvector ret = *this;
    ret.v -= rhs.v;
    return ret;
  }
  fvector operator-()
  {
    fvector ret = *this;
    ret.v = -ret.v;
    return ret;
  }

  fvector operator!() const
  {
    fvector ret = *this;
    for(auto& v : ret.a)
      v = !v;

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
    for(unsigned int i = 0 ; i < W; ++i)
      ret.a[i] = a[i] < rhs.a[i];
    return ret;
  }

  bool operator==(const fvector& rhs) const
  {
    return std::equal(a.cbegin(), a.cend(), rhs.a.cbegin(), rhs.a.cend());
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


template<unsigned int W>
auto exp(const fvector<W>& v)
{
  fvector<W> ret;
  for(size_t i = 0; i < W ; ++i) {
    ret.a[i] = expf(v.a[i]);
  }
  return ret;
}

template<unsigned int W>
auto log(const fvector<W>& v)
{
  fvector<W> ret;
  for(size_t i = 0; i < W ; ++i) {
    ret.a[i] = logf(v.a[i]);
  }
  return ret;
}


template<unsigned int W>
std::ostream& operator<<(std::ostream& os, const fvector<W>& o) 
{
  for(const auto& el : o.a)
    os << el <<" ";
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
    ret.a[i] = std::max(lhs.a[i], rhs.a[i]);
  return ret;
}

