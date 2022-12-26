#pragma once
#include "array.hh"
#include <unistd.h>
#include <iostream>

template<unsigned int IN, unsigned int OUT>
struct Linear
{
  NNArray<float, OUT, IN> d_weights;
  NNArray<float, OUT, 1> d_bias;

  Linear()
  {
    randomize();
  }
  void randomize() // "Xavier initialization"  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  {
    d_weights.randomize(1.0/sqrt(IN));
    d_bias.randomize(1.0/sqrt(IN));
  }

  auto forward(const NNArray<float, IN, 1>& in)
  {
    return d_weights * in + d_bias;
  }

  void learn(float lr)
  {
    auto grad1 = d_weights.getGrad();
    grad1 *= lr;
    d_weights -= grad1;
    
    auto grad2 = d_bias.getGrad();
    grad2 *= lr;
    d_bias -= grad2;
  }

  void save(std::ostream& out) const
  {
    d_weights.save(out);
    d_bias.save(out);
  }
  void load(std::istream& in)
  {
    d_weights.load(in);
    d_bias.load(in);
  }
};


template<unsigned int ROWS, unsigned int COLS, unsigned int KERNEL,
         unsigned int INLAYERS, unsigned int OUTLAYERS>
struct Conv2d
{
  std::array<NNArray<float, KERNEL, KERNEL>, OUTLAYERS> d_filters;
  std::array<NNArray<float, 1, 1>, OUTLAYERS> d_bias;

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

  auto forward(NNArray<float, ROWS, COLS>& in)
  {
    std::array<NNArray<float, ROWS, COLS>, 1> a;
    a[0] = in;
    return forward(a);
  }
  
  auto forward(std::array<NNArray<float, ROWS, COLS>, INLAYERS>& in)
  {
    std::array<NNArray<float, 1+ROWS-KERNEL, 1 + COLS - KERNEL>, OUTLAYERS> ret;

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

  void learn(float lr)
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

  void save(std::ostream& out) const
  {
    for(const auto& w : d_filters)
      w.save(out);
    for(const auto& w : d_bias)
      w.save(out);
  }
  void load(std::istream& in)
  {
    for(auto& w : d_filters)
      w.load(in);
    for(auto& w : d_bias)
      w.load(in);
  }
};

template<unsigned int ROWS, unsigned int COLS, long unsigned int CHANNELS>
auto flatten(const std::array<NNArray<float, ROWS, COLS>, CHANNELS>& in)
{
  NNArray<float, ROWS*COLS*CHANNELS, 1> ret;
  
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
