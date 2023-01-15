#pragma once
#include <vector>
#include <random>
#include "tracked.hh"

template<typename T, unsigned int ROWS, unsigned int COLS>
struct SArray
{
  typedef SArray<T, ROWS, COLS> us_t;
  SArray()
  {
    d_store.resize(ROWS*COLS);
  }

  void setZero()
  {
    for(auto& v : d_store)
      v = 0;
  }

  auto getCols() const
  {
    return COLS;
  }
  auto getRows() const
  {
    return ROWS;
  }

  constexpr auto size() const
  {
    return ROWS * COLS;
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

  auto& operator+=(const SArray<T, ROWS, COLS>& rhs)
  {
    for(size_t pos = 0 ; pos < rhs.d_store.size(); ++pos)
      d_store[pos] += rhs.d_store[pos];
    return *this;
  }

  auto operator+(const SArray<T, ROWS, COLS>& rhs) const
  {
    us_t ret = *this;
    return ret += rhs;
  }

  
  auto operator/=(float val)
  {
    for(auto& v : d_store)
      v/=val;
    return *this;
  }

  auto operator/(float val) const
  {
    SArray<T, ROWS, COLS> ret = *this;
    for(auto& v : ret.d_store)
      v/=val;
    return ret;
  }

  auto operator*=(float val)
  {
    for(auto& v : d_store)
      v*=val;
    return *this;
  }

  auto operator*(float val) const
  {
    auto ret = *this;
    for(auto& v : ret.d_store)
      v *= val;
    return ret;
  }

  auto operator==(const SArray<T, ROWS, COLS>& rhs) const
  {
    return std::equal(d_store.cbegin(), d_store.cend(), rhs.d_store.cbegin(), rhs.d_store.cend());
  }
  // if we carry a vector type for SSE2/AVX etc, this will add up all the elements per element
  auto unparallel() const 
  {
    SArray<float, ROWS, COLS> ret;
    for(size_t pos = 0 ; pos < ret.d_store.size(); ++pos) {
      ret.d_store[pos] = d_store[pos].sum();
    }
    return ret;
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
  typedef NNArray<T, ROWS, COLS> us_t;
  typedef SArray<T, ROWS, COLS> sus_t; // very
  
  TrackedNumber<T>& operator()(int x, int y)
  {
    return d_store.at(x*COLS + y);
  }
  
  const TrackedNumber<T>& operator()(int x, int y) const
  {
    return d_store.at(x*COLS + y);
  }

  constexpr auto getCols() const
  {
    return COLS;
  }
  constexpr auto getRows() const
  {
    return ROWS;
  }

  static constexpr unsigned int SIZE = ROWS*COLS;
  constexpr auto size() const
  {
    return ROWS * COLS;
  }

  auto getS() const
  {
    sus_t ret;
    for(size_t pos = 0; pos < d_store.size(); ++pos)
      ret.d_store[pos] = d_store[pos].getVal();
    return ret;
  }
  auto getGrad() const
  {
    SArray<T, ROWS, COLS> ret;
    ret.d_store.resize(d_store.size());
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      ret.d_store[pos] = d_store[pos].getGrad();
    return ret;
  }

  void addGrad(const SArray<T, ROWS, COLS>& rhs) 
  {
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      d_store[pos].impl->d_grad += rhs.d_store[pos];
  }

  void setGrad(const SArray<T, ROWS, COLS>& rhs) 
  {
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      d_store[pos].impl->d_grad = rhs.d_store[pos];
  }

  void setGradCons(const SArray<T, ROWS, COLS>& rhs) 
  {
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      d_store[pos].impl->d_grad = rhs.d_store[pos]; //.sum();
  }

  void setVariable()
  {
    for(auto& v : d_store)
      v.setVariable();
  }
  void needsGrad()
  {
    for(auto& v : d_store)
      v.needsGrad();
  }
  
  // hadamard
  auto dot(const us_t& rhs)
  {
    us_t ret;
    for(size_t pos = 0 ; pos < d_store.size(); ++pos) {
      ret.d_store[pos] = d_store[pos] * rhs.d_store[pos];
    }
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

  template<class Q=T,
           class std::enable_if<std::is_union<Q>::value, int>::type = 0>
  auto& operator-=(const SArray<float, ROWS, COLS>& rhs)
  //  auto& decrUnparallel(const SArray<float, ROWS, COLS>& rhs)
  {
    // this changes the contents of weights to a new numerical value, based on the old one
    // by doing it like this, tracking is retained
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      d_store[pos] = d_store[pos].getVal() - rhs.d_store[pos];

    return *this;
  }

  void randomize(T fact=1.0)
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};

    for(auto& item : d_store) {
      item = (float)d(gen)*fact;
      item.setVariable();
    }
  }

  // wipes out all history
  void reset()
  {
    for(auto& v : d_store)
      v.impl.reset();
    zero();
  }
  
  void zero()
  {
    constant(0);
  }
  void constant(T val)
  {
    for(auto& item : d_store) {
      item = val;
    }
  }

  template<typename F>
  auto applyFunc([[maybe_unused]] const F& f)
  {
    NNArray<T, ROWS, COLS> ret;
    
    ret.d_store.clear(); 
    for(const auto& v : d_store)
      ret.d_store.push_back(makeFunc(v, f));
    return ret;
  }

  // does it in ALL fields, not by row
  auto norm() 
  {
    NNArray<T, ROWS, COLS> ret;
    TrackedNumber<T> sum;
    bool first = true;
    for(const auto& v : d_store) {
      if(first) {
        sum = v;
        first=false;
      }
      sum = sum + v;
    }
    for(unsigned int pos = 0 ; pos < ret.d_store.size() ; ++pos)
      ret.d_store[pos] = d_store[pos]/sum;
    return ret;
  }

  
  // does it in ALL fields, not by row
  // https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
  auto logSoftMax() 
  {
    NNArray<T, ROWS, COLS> ret;

    TrackedNumber<T> lemax=d_store.at(0);
    for(size_t pos = 1; pos < d_store.size(); ++pos)
      lemax = makeMax(lemax, d_store[pos]);

    TrackedNumber<T> sum;
    bool first=true;
    for(const auto& v : d_store) {
      if(first) {
        sum = makeFunc(v - lemax, ExpFunc());
        first = false;
      }
      else 
        sum = sum + makeFunc(v - lemax, ExpFunc());
    }

    TrackedNumber<T> logsum = makeFunc(sum, LogFunc());
    for(unsigned int pos = 0 ; pos < ret.d_store.size() ; ++pos)
      ret.d_store[pos] = d_store[pos] - lemax - logsum;
    return ret;
  }
  
  auto flatViewRow() const
  {
    NNArray<T, ROWS*COLS, 1> ret;
    ret.d_store = d_store;
    return ret;
  }
  auto flatViewCol() const
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

  TrackedNumber<T> mean()
  {
    return sum() / TrackedNumber<T>((float)d_store.size());
  }

  auto getMeanStd() // numerical recipes 14.1
  {
    std::pair<T, T> ret{0,0};
    for(auto& item : d_store) {
      ret.first += item.getVal();
    }
    ret.first /= d_store.size(); // have mean now
    T diffsum=0, diff2sum=0;
    for(auto& item : d_store) {
      auto diff= (item.getVal() - ret.first);
      diff2sum += diff*diff;
      diffsum += diff;

    }
    diffsum *= diffsum;
    diffsum /= d_store.size();
    ret.second = sqrt( (diff2sum - diffsum) / (d_store.size() -1));
    return ret; 
  }

  auto getUnparallel(unsigned int idx)
  {
    NNArray<float, ROWS, COLS> ret;
    for(unsigned int i = 0 ; i < ret.d_store.size(); ++i) {
      ret.d_store[i] = d_store[i].getVal().v[idx];
    }
    return ret;
  }
  
  // goes down a column to find the row with the x-est value
  unsigned int xValueIndexOfColumn(int col, float fact)
  {
    float xval=fact*(*this)(0, col).getVal();
    int xrow=0;
    for(unsigned int r=0; r < ROWS; ++r) {
      float val = fact*(*this)(r, col).getVal();
      //      std::cout<<"ROWS " <<ROWS<< " r "<<r<<  " col " << col << " val " <<val<< " xval "<<xval << " xrow "<< xrow<<std::endl;
      if(val > xval) {
        xval = val;
        xrow=r;
      }
    }
    return xrow;
  }

  // goes down a column to find the row with the highest value
  unsigned int maxValueIndexOfColumn(int col)
  {
    return xValueIndexOfColumn(col, 1.0);
  }
  // goes down a column to find the row with the highest value
  unsigned int minValueIndexOfColumn(int col)
  {
    return xValueIndexOfColumn(col, -1.0);
  }

  template<unsigned int N>
  NNArray<T, ROWS, N>
  operator*(const NNArray<T, COLS, N>& a) const
  {
    NNArray<T, ROWS, N> ret;

    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ret(i,j) = 0;
        for (size_t k = 0; k < COLS; ++k) {
          ret(i,j) = ret(i,j) + (*this)(i,k) * a(k,j);
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

  auto elMult(NNArray<T, ROWS, COLS>& w)
  {
    NNArray<T, ROWS, COLS> ret;
    for(size_t pos = 0 ; pos < d_store.size(); ++pos)
      ret.d_store[pos] = d_store[pos] * w.d_store[pos];
    return ret;
  }
  
  template<unsigned int KERNEL>
  NNArray<T, 1+ROWS-KERNEL, 1+COLS-KERNEL>
  Convo2d(NNArray<T, KERNEL, KERNEL>& weights, NNArray<T, 1, 1>& bias)
  {
    NNArray<T, 1+ROWS-KERNEL, 1+COLS-KERNEL> ret;
    NNArray<T, KERNEL, KERNEL> kernel;

    for(unsigned int r=0; r < 1+ROWS-KERNEL; ++r) {
      for(unsigned int c=0; c < 1+COLS-KERNEL; ++c) {
        for(unsigned int kr=0; kr < KERNEL; ++kr) {
          for(unsigned int kc=0; kc < KERNEL; ++kc) { 
            kernel(kr,kc) = (*this)(r + kr, c + kc);
          }
        }
        ret(r,c) = kernel.elMult(weights).sum() + bias(0,0);
      }
    }
    return ret;
  }

  template<unsigned int KERNEL>
  
  auto Max2d()
  {
    // this is for padding..
    NNArray<T, (ROWS+KERNEL-1)/KERNEL, (COLS+KERNEL-1)/KERNEL> ret;
    NNArray<T, KERNEL, KERNEL> kernel;

    for(unsigned int r=0; r < (ROWS+KERNEL-1)/KERNEL; ++r) {
      for(unsigned int c=0; c < (COLS+KERNEL-1)/KERNEL; ++c) {
        // this will not require padding, is leftmost element
        TrackedNumber<T> max = (*this)(r*KERNEL, c*KERNEL);
        for(unsigned int kr=0; kr < KERNEL; ++kr) {
          for(unsigned int kc=0; kc < KERNEL; ++kc) {
            if(r*KERNEL + kr < ROWS && c*KERNEL +kc < COLS)
              max = makeMax(max, (*this)(r*KERNEL+kr,c*KERNEL+kc));
            // "do nothing" if we are beyond the edge of the input
          }
        }
        ret(r,c) = max;
      }
    }
    
    return ret;
  }

  static float extr(float in) 
  {
    return in;
  }
  static float extr(const fvector<4>& in) 
  {
    return in.v[0];
  }
  
  static float extr(const fvector<8>& in) 
  {
    return in.v[0];
  }
  
  void save(std::ostream& out) const
  {
    float rows=ROWS, cols=COLS;
    auto swrite = [&out](float v) {
      out.write((char*)&v, sizeof(v));
    };
    swrite(rows);
    swrite(cols);
    for(const auto& v : d_store)
      swrite(extr(v.getVal()));
  }

  void load(std::istream& in)
  {
    auto sread = [&in]() {
      float v;
      in.read((char*)&v, sizeof(v));
      return v;
    };
    if(ROWS != sread() || COLS !=sread())  // living dangerously here!
      throw std::logic_error("Dimensions of stream to load from do not match");
    
    for(auto& v : d_store)
      v = sread();
  }
};



template<typename T, unsigned int ROWS, unsigned int COLS>
NNArray<T, ROWS, COLS> operator-(const NNArray<T, ROWS, COLS>& lhs, const NNArray<T, ROWS, COLS>& rhs)
{
  NNArray<T, ROWS, COLS> ret;
  for(size_t pos = 0 ; pos < lhs.d_store.size(); ++pos)
    ret.d_store[pos] = lhs.d_store[pos]  - rhs.d_store[pos];
  
  return ret;
}

template<typename T, unsigned int ROWS, unsigned int COLS>
NNArray<T, ROWS, COLS> operator+(const NNArray<T, ROWS, COLS>& lhs, const NNArray<T, ROWS, COLS>& rhs)
{
  NNArray<T, ROWS, COLS> ret;
  for(size_t pos = 0 ; pos < lhs.d_store.size(); ++pos)
    ret.d_store[pos] = lhs.d_store[pos] + rhs.d_store[pos];
  
  return ret;
}

template<typename T, unsigned int ROWS, unsigned int COLS>
std::ostream& operator<<(std::ostream& os, const NNArray<T, ROWS, COLS>& ns)
{
  for(size_t r = 0; r < ROWS; ++r) {
    for(size_t c = 0; c < COLS; ++c) {
      if(c)
        os<<' ';
      os<< ns(r,c).getVal();
    }
    os<<'\n';
  }
      

  return os;
}


template<typename T, unsigned int ROWS, unsigned int COLS>
std::ostream& operator<<(std::ostream& os, const SArray<T, ROWS, COLS>& ns)
{
  for(size_t r = 0; r < ROWS; ++r) {
    for(size_t c = 0; c < COLS; ++c) {
      if(c)
        os<<' ';
      os<< ns(r,c);
    }
    os<<'\n';
  }
      

  return os;
}

template<typename A, typename W>
inline auto makeProj(const A& arr, const W& w)
{
  // goes over the store, records per pointer what position it is at
  std::array<unsigned int, arr.SIZE> proj;
  typename decltype(w.dyns)::value_type rt;
  std::unordered_map<decltype(rt.second), decltype(rt.first)> rev;
  for(const auto& p : w.dyns)
    rev[p.second] = p.first;
  
  for(size_t pos = 0 ; pos < arr.SIZE; ++pos) {
    if(auto iter = rev.find(arr.d_store[pos].impl.get()) ; iter != rev.end())
      proj[pos] = iter->second;
    else
      proj[pos] = std::numeric_limits<unsigned int>::max();
  }
  return proj;
}

template<typename PROJ, typename SRC, typename DST>
inline void projForward(const PROJ& proj, const SRC& src, DST& dst)
{
  static_assert(std::tuple_size<PROJ>::value == src.SIZE);
  size_t pos = 0;
  
  for(const auto& v : src.d_store) {
    if(proj[pos] != std::numeric_limits<unsigned int>::max()) // not everything needs to be mapped forward
      dst.work[proj[pos++]].ourval = v.getVal();
  }
}

template<typename PROJ, typename SRC, typename DST>
inline void projBack(const PROJ& proj, const SRC& src, DST& dst)
{
  static_assert(std::tuple_size<PROJ>::value == dst.SIZE);
  size_t pos = 0;
  for(auto& v : dst.d_store) {
    v = src.work[proj[pos]].ourval;
    v.impl->d_grad = src.grads[proj[pos]];
    pos++;
  }
}

template<typename PROJ, typename SRC, typename DST>
inline void projBackGrad(const PROJ& proj, const SRC& src, DST& dst)
{
  static_assert(std::tuple_size<PROJ>::value == dst.SIZE);
  size_t pos = 0;
  for(auto& v : dst.d_store) {
    v.impl->d_grad += src.grads[proj[pos++]].sum();
  }
}
