#pragma once
#include "tensor2.hh"
#include <unistd.h>
#include <fstream>
#include <vector>
#include "ext/sqlitewriter/sqlwriter.hh"

template<typename T>
struct TensorLayer
{
  TensorLayer() {} 
  TensorLayer(const TensorLayer& rhs) = delete;
  TensorLayer& operator=(const TensorLayer&) = delete;
  
  virtual void randomize() = 0;
  void save(std::ostream& out) const
  {
    for(const auto& p : d_params)
      p.ptr->save(out);
  }
  void load(std::istream& in) 
  {
    for(auto& p : d_params)
      p.ptr->load(in);
  }
  void learn(float lr, float momentum) 
  {
    for(auto& p : d_params) {
      auto grad1 = (momentum * p.ptr->getAccumGrad()).eval(); // THIS IS THE MOMENTUM
      if(p.ptr->getPrevAccumGrad().rows())
        grad1 += p.ptr->getPrevAccumGrad();
      grad1 *= lr;
      *p.ptr -= grad1;
    }
  }
  // emit all parameters to SQL
  void emit(SQLiteWriter& sqw, unsigned int startID, unsigned int& pos, unsigned int batchno, unsigned int batchsize, std::string layername)
  {
    auto emit = [&sqw, &startID, &pos, &batchno, &batchsize](const auto& t, std::string_view name, std::string_view subname, unsigned int idx) {
      for(unsigned int r = 0 ; r < t.d_imp->d_val.rows(); ++r) {
        for(unsigned int c = 0 ; c < t.d_imp->d_val.cols(); ++c) {
          sqw.addValue({
              {"batchno", batchno},
              {"pos", pos},
              {"val", t.d_imp->d_val(r,c)},
              {"grad", t.d_imp->d_accumgrads(r,c)/batchsize},
              {"idx", idx},
              {"row", r},
              {"col", c},
              {"name", (std::string)name},
              {"subname", (std::string)subname},
              {"startID", startID}});
          ++pos;
        }
      }
    };
    // iterate over params
    for(const auto& p : d_params)
      emit(*p.ptr, layername, p.subname, p.idx);
  }

  struct Param
  {
    Tensor<T>* ptr;
    std::string subname;
    unsigned int idx=0;
  };
  std::vector<Param> d_params;
};

template<typename T, unsigned int IN, unsigned int OUT>
struct Linear : public TensorLayer<T>
{
  Tensor<T> d_weights{OUT, IN};
  Tensor<T> d_bias{OUT, 1};
  
  Linear()
  {
    randomize();
    this->d_params = {{&d_weights, "weights"},  {&d_bias, "bias"}};
  }
  void randomize() override// "Xavier initialization"  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  {
    d_weights.randomize(1.0/sqrt(IN));
    d_bias.randomize(1.0/sqrt(IN));
  }

  auto forward(const Tensor<T>& in)
  {
    return d_weights * in + d_bias;
  }

  auto SquaredWeightsSum()
  {
    return makeFunction<SquareFunc>(d_weights).sum();
  }

};


template<typename T, unsigned int ROWS, unsigned int COLS, unsigned int KERNEL,
         unsigned int INLAYERS, unsigned int OUTLAYERS>
struct Conv2d : TensorLayer<T>
{
  std::array<Tensor<T>, OUTLAYERS> d_filters;
  std::array<Tensor<T>, OUTLAYERS> d_bias;

  Conv2d()
  {
    unsigned int idx=0;
    for(auto& f : d_filters) {
      f = Tensor<T>(KERNEL, KERNEL);
      this->d_params.push_back({&f, "filter", idx});
      ++idx;
    }
    idx=0;
    for(auto& b : d_bias) {
      b = Tensor<T>(1,1);
      this->d_params.push_back({&b, "bias", idx});
      ++idx;
    }
        
    randomize();
  }

  void randomize()
  {
    for(auto& f : d_filters) {
      f.randomize(sqrt(1.0/(INLAYERS*KERNEL*KERNEL)));
    }
    for(auto& b : d_bias) {
      b.randomize(sqrt(1.0/(INLAYERS*KERNEL*KERNEL)));
    }
  }

  
  auto forward(Tensor<T>& in)
  {
    std::array<Tensor<T>, 1> a;
    a[0] = in;
    return forward(a);
  }
  
  auto forward(std::array<Tensor<T>, INLAYERS>& in)
  {
    std::array<Tensor<T>, OUTLAYERS> ret;

    for(auto& o : ret)
      o = Tensor<T>(1+ROWS-KERNEL, 1 + COLS - KERNEL);
    
    // The output layers of the next convo2d have OUT filters
    // these filters need to be applied to all IN input layers
    // and the output is the addition of the outputs of those filters
    
    unsigned int ctr = 0;
    for(auto& p : ret) { // outlayers long
      p.zero();
      for(auto& p2 : in)
        p = p +  p2.makeConvo(KERNEL, d_filters.at(ctr), d_bias.at(ctr));
      ctr++;
    }
    return ret;
  }

  auto SquaredWeightsSum()
  {
    Tensor<T> ret(1,1);
    ret(0,0)=0;
    for(auto& f : d_filters) 
      ret = ret + makeFunction<SquareFunc>(f).sum();
    return ret;
  }
};

template<typename T>
struct ModelState
{
  ModelState() {}
  ModelState(const ModelState&) = delete;
  ModelState& operator=(const ModelState& rhs) = delete;
  void randomize()
  {
    for(auto& m : d_members)
      m.first->randomize();
  }
  void learn(float lr, float momentum=0)
  {
    for(auto& m : d_members)
      m.first->learn(lr, momentum);
  }

  void save(std::ostream& out) const
  {
    for(const auto& m : d_members)
      m.first->save(out);
  }
  void load(std::istream& in)
  {
    for(auto& m : d_members)
      m.first->load(in);
  }

  std::vector<std::pair<TensorLayer<T>*, std::string>> d_members;

  // ask all members of a model to emit themselves for logging
  void emit(SQLiteWriter& sqw, unsigned int startID, unsigned int batchno, unsigned int batchsize)
  {
    unsigned int pos = 0;
    for(const auto& m : d_members) {
      m.first->emit(sqw, startID, pos, batchno, batchsize, m.second);
    }
  }
};


template<typename MS>
void saveModelState(const MS& ms, const std::string& fname)
{
  std::ofstream ofs(fname+".tmp");
  if(!ofs)
    throw std::runtime_error("Can't save model to file "+fname+".tmp: "+strerror(errno));
  std::array<float, 2>v={12345.0, 67890.0}; // magic begin
  ofs.write((const char*)&v[0], 2*sizeof(float));
  ms.save(ofs);
  v={9876.0,54321.0}; // magic end
  ofs.write((const char*)&v[0], 2*sizeof(float));
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
  float v[2]={};
  ifs.read((char*)v, 2*sizeof(float));
  if(v[0] != 12345 || v[1] != 67890)
    throw std::runtime_error("Model state has wrong begin magic, "+std::to_string(v[0])+", "+std::to_string(v[1]));
  ms.load(ifs);
  ifs.read((char*)v, 2*sizeof(float));
  if(v[0] != 9876 || v[1] != 54321)
    throw std::runtime_error("Model state has wrong end magic, "+std::to_string(v[0])+", "+std::to_string(v[1]));

}
