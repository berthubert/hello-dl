#pragma once
#include "array.hh"
#include <unistd.h>
#include <iostream>

struct LayerBase
{
  virtual void save(std::ostream& out) const = 0;
  virtual void load(std::istream& in)=0;
  virtual void learn(float lr) = 0;
  virtual void zeroGrad() = 0;
  virtual unsigned int size() const = 0;
};

template<typename T, unsigned int IN, unsigned int OUT>
struct Linear : LayerBase
{
  NNArray<T, OUT, IN> d_weights;
  NNArray<T, OUT, 1> d_bias;

  Linear()
  {
    randomize();
  }
  void randomize() // "Xavier initialization"  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  {
    d_weights.randomize(1.0/sqrt(IN));
    d_bias.randomize(1.0/sqrt(IN));
  }

  unsigned int size() const override
  {
    return d_weights.size() + d_bias.size();
  }
  
  void zeroGrad() override
  {
    d_weights.zeroGrad();
    d_bias.zeroGrad();
  }

  void addGrad(const Linear& rhs)
  {
    d_weights.addGrad(rhs.d_weights.getGrad());
    d_bias.addGrad(rhs.d_bias.getGrad());
  }

  void setGrad(const Linear& rhs, float divisor)
  {
    d_weights.setGradCons(rhs.d_weights.getGrad()/divisor);
    d_bias.setGradCons(rhs.d_bias.getGrad()/divisor);
  }

  
  auto forward(const NNArray<T, IN, 1>& in)
  {
    return d_weights * in + d_bias;
  }

  void learn(float lr) override
  {
    auto grad1 = d_weights.getGrad();
    grad1 *= lr;
    d_weights -= grad1;
    
    auto grad2 = d_bias.getGrad();
    grad2 *= lr;
    d_bias -= grad2;
  }

  void save(std::ostream& out) const override
  {
    d_weights.save(out);
    d_bias.save(out);
  }
  void load(std::istream& in) override
  {
    d_weights.load(in);
    d_bias.load(in);
  }
};


template<typename T, unsigned int ROWS, unsigned int COLS, unsigned int KERNEL,
         unsigned int INLAYERS, unsigned int OUTLAYERS>
struct Conv2d : LayerBase
{
  std::array<NNArray<T, KERNEL, KERNEL>, OUTLAYERS> d_filters;
  std::array<NNArray<T, 1, 1>, OUTLAYERS> d_bias;

  Conv2d()
  {
    randomize();
  }

  void randomize()
  {
    for(auto& f : d_filters)
      f.randomize(sqrt(1.0/(INLAYERS*KERNEL*KERNEL)));
    for(auto& b : d_bias)
      b.randomize(sqrt(1.0/(INLAYERS*KERNEL*KERNEL)));
  }

  unsigned int size() const override
  {
    unsigned int ret = 0;
    for(auto& f : d_filters)
      ret += f.size();
    for(auto& b : d_bias)
      ret += b.size();
    return ret;
  }
  
  
  void zeroGrad() override
  {
    for(auto& f : d_filters)
      f.zeroGrad();
    for(auto& b : d_bias)
      b.zeroGrad();
  }

  void addGrad(const Conv2d& rhs)
  {
    for(size_t i = 0 ; i < d_filters.size(); ++i)
      d_filters[i].addGrad(rhs.d_filters[i].getGrad());
    for(size_t i = 0 ; i < d_bias.size(); ++i)
      d_bias[i].addGrad(rhs.d_bias[i].getGrad());
  }

  void setGrad(const Conv2d& rhs, float divisor)
  {
    for(size_t i = 0 ; i < d_filters.size(); ++i)
      d_filters[i].setGradCons(rhs.d_filters[i].getGrad()/divisor);
    for(size_t i = 0 ; i < d_bias.size(); ++i)
      d_bias[i].setGradCons(rhs.d_bias[i].getGrad()/divisor);
  }

  
  
  auto forward(NNArray<T, ROWS, COLS>& in)
  {
    std::array<NNArray<T, ROWS, COLS>, 1> a;
    a[0] = in;
    return forward(a);
  }
  
  auto forward(std::array<NNArray<T, ROWS, COLS>, INLAYERS>& in)
  {
    std::array<NNArray<T, 1+ROWS-KERNEL, 1 + COLS - KERNEL>, OUTLAYERS> ret;

    // The output layers of the next convo2d have OUT filters
    // these filters need to be applied to all IN input layers
    // and the output is the addition of the outputs of those filters
    
    unsigned int ctr = 0;
    for(auto& p : ret) {
      p.zero();
      for(auto& p2 : in)
        p = p +  p2. template Convo2d<KERNEL>(d_filters.at(ctr), d_bias.at(ctr));
      ctr++;
    }
    return ret;
  }

  void learn(float lr) override
  {
    for(auto& v : d_filters) {
      auto grad = v.getGrad();
      grad *= lr;
      v -= grad;
    }
    for(auto& v : d_bias) {
      auto grad = v.getGrad();
      grad *= lr;
      v -= grad;
    }
  }

  void save(std::ostream& out) const override
  {
    for(const auto& w : d_filters)
      w.save(out);
    for(const auto& w : d_bias)
      w.save(out);
  }
  void load(std::istream& in) override
  {
    for(auto& w : d_filters)
      w.load(in);
    for(auto& w : d_bias)
      w.load(in);
  }
};

template<typename T, unsigned int ROWS, unsigned int COLS, long unsigned int CHANNELS>
auto flatten(const std::array<NNArray<T, ROWS, COLS>, CHANNELS>& in)
{
  NNArray<T, ROWS*COLS*CHANNELS, 1> ret;
  
  unsigned int ctr=0;
  for(const auto& p : in) {
    auto flatpart = p.flatViewRow();
    for(unsigned int i = 0; i < p.getRows() * p.getCols(); ++i) 
      ret(ctr++, 0) = flatpart(i, 0);
  }
  return ret;
}

template<typename MS>
void saveModelState(const MS& ms, const std::string& fname)
{
  std::ofstream ofs(fname+".tmp");
  if(!ofs)
    throw std::runtime_error("Can't save model to file "+fname+".tmp: "+strerror(errno));
  ms.save(ofs);
  ofs.flush();
  ofs.close();
  unlink(fname.c_str());
  rename((fname+".tmp").c_str(), fname.c_str());
}

template<typename MS>
void loadModelState(MS& ms, const std::string& fname)
{
  std::ifstream ifs(fname);
  if(!ifs)
    throw std::runtime_error("Can't read model state from file "+fname+": "+strerror(errno));
  ms.load(ifs);
}
