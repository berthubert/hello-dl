#pragma once
#include <vector>
#include <random>
#include "tracked.hh"

template<typename T, unsigned int ROWS, unsigned int COLS>
struct SArray
{
  SArray()
  {
    d_store.resize(ROWS*COLS);
  }
  std::vector<T> d_store;
  T& operator()(int x, int y)
  {
    return d_store.at(x*COLS + y);
  }
  const T& operator()(int x, int y) const
  {
    return d_store.at(x*COLS + y);
  }

  auto operator+=(const SArray<T, ROWS, COLS>& rhs)
  {
    for(size_t pos = 0 ; pos < rhs.d_store.size(); ++pos)
      d_store[pos] += rhs.d_store[pos];
    return *this;
  }

  auto operator/=(float val)
  {
    for(auto& v : d_store)
      v/=val;
    return *this;
  }
  auto operator*=(float val)
  {
    for(auto& v : d_store)
      v*=val;
    return *this;
  }
  
};


template<typename T, unsigned int ROWS, unsigned int COLS>
struct NNArray
{
  NNArray()
  {
    d_store.resize(ROWS*COLS);
  }
  std::vector<TrackedNumber<T>> d_store;
  TrackedNumber<T>& operator()(int x, int y)
  {
    return d_store.at(x*COLS + y);
  }


  const TrackedNumber<T>& operator()(int x, int y) const
  {
    return d_store.at(x*COLS + y);
  }

  auto getGrad() const
  {
    SArray<T, ROWS, COLS> ret;
    ret.d_store.resize(d_store.size());
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      ret.d_store[pos] = d_store[pos].getGrad();
    return ret;
  }

  auto& operator-=(const SArray<T, ROWS, COLS>& rhs)
  {
    // this changes the contents of weights to a new numerical value, based on the old one
    // by doing it like this, tracking is retained
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      d_store[pos] = d_store[pos].getVal() - rhs.d_store[pos];

    return *this;
  }

  auto& operator+(const NNArray<T, ROWS, COLS>& rhs)
  {
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      d_store[pos] = d_store[pos]  + rhs.d_store[pos];
    
    return *this;
  }
  
  void randomize()
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};

    for(auto& item : d_store) {
      item = (float)d(gen);
    }
  }

  auto flatViewRow()
  {
    NNArray<T, ROWS*COLS, 1> ret;
    ret.d_store = d_store;
    return ret;
  }
  auto flatViewCol()
  {
    NNArray<T, 1, ROWS*COLS> ret;
    ret.d_store = d_store;
    return ret;
  }
  
  TrackedNumber<T> sum()
  {
    TrackedNumber<T> ret{0};
    for(auto& item : d_store) {
      ret = ret + item;
    }
    return ret;
  }

  // *this is ROWS*COLS
  // a is COLS*N
  
  template<unsigned int N>
  NNArray<T, ROWS, N>
  operator*(const NNArray<T, COLS, N>& a) const
  {
    NNArray<T, ROWS, N> ret;
    //    std::cout << "N " <<N << " ROWS " << ROWS <<" COLS "<<COLS<<std::endl;

    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ret(i,j) = 0;
        for (size_t k = 0; k < COLS; ++k) {
          //          std::cout << i<<","<<j<<","<<k<< std::endl;
          // XXX bit worried over the below
          ret(i,j) = ret(i,j) + (*this)(i,k) * a(k,j);
          //std::cout<< ret(i,j).getVal() << std::endl;
        }
      }
    }
    
    return ret;
  }

  void zeroGrad()
  {
    for(auto& v : d_store)
      v.zeroGrad();
  }
  
};


