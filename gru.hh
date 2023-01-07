#pragma once
#include "layers.hh"


// hidden state=> dense linear => output x

// x is input
// h_t, h_{t-1} = hidden state

// gate_{reset} = \sigma(W_{input_{reset}} \cdot x_t + W_{hidden_{reset}} \cdot h_{t-1})

// W_input_reset  - ^^ normal matrix products
// W_input_hidden

// pytorch:

// r_t​ = σ(W_{ir} ​x_t + b_{ir}​+W_{hr}​h_{t−1}​ +b_{hr}​)            // reset gate
// z_t​ = σ(W_{iz} ​x_t ​+ b_{iz} ​+W_{hz}​h_{t−1}​ + b_{hz}​)          // update
// n_t​ = tanh(W_{in}​x_t​+b_{in}​+ r_t​*(W_{hn} ​h_{t−1}​ + b_{hn}​))   // "new" - * is dotproduct
// h_t​=(1−z_t​)*n_t​+z_t​*h_{t−1)​}                                  // new h

// the hidden state is also the output, which needs linear combination to turn into input size again
// https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
template<typename T, unsigned int IN, unsigned int HIDDEN>
struct GRULayer : LayerBase
{
  NNArray<T, HIDDEN, IN> d_w_ir; // reset
  NNArray<T, HIDDEN, IN> d_w_iz; // update
  NNArray<T, HIDDEN, IN> d_w_in; // new

  NNArray<T, HIDDEN, HIDDEN> d_w_hr;  // hidden reset
  NNArray<T, HIDDEN, HIDDEN> d_w_hz; // hidden update
  NNArray<T, HIDDEN, HIDDEN> d_w_hn; // hidden "new"

  NNArray<T, HIDDEN, 1> d_prevh;

  std::array<unsigned int, decltype(d_w_ir)::SIZE> d_w_ir_proj;
  std::array<unsigned int, decltype(d_w_iz)::SIZE> d_w_iz_proj;
  std::array<unsigned int, decltype(d_w_in)::SIZE> d_w_in_proj;

  std::array<unsigned int, decltype(d_w_hr)::SIZE> d_w_hr_proj;
  std::array<unsigned int, decltype(d_w_hz)::SIZE> d_w_hz_proj;
  std::array<unsigned int, decltype(d_w_hn)::SIZE> d_w_hn_proj;

  GRULayer()
  {
    randomize();
  }

  // https://blog.floydhub.com/gru-with-pytorch/
  //  https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18
  // these appear to be slightly different
  auto forward(const NNArray<T, IN, 1>& xt)
  {
    auto r_t = (d_w_ir * xt + d_w_hr * d_prevh).applyFunc(SigmoidFunc()); // reset gate
    auto z_t = (d_w_iz * xt + d_w_hz * d_prevh).applyFunc(SigmoidFunc());
    auto n_t = (d_w_in * xt + r_t.dot(d_w_hn *d_prevh)).applyFunc(TanhFunc());
    NNArray<T, HIDDEN, 1> one;
    one.constant(1.0);
    auto h_t = (one - z_t).dot(n_t) + z_t.dot(d_prevh);
    d_prevh = h_t; // "this is where the magic happens"
    return h_t;
  }

  void learn(float lr) override
  {
    { auto grad1 = d_w_ir.getGrad(); grad1 *= lr; d_w_ir -= grad1;  }
    { auto grad1 = d_w_iz.getGrad(); grad1 *= lr; d_w_iz -= grad1;  }
    { auto grad1 = d_w_in.getGrad(); grad1 *= lr; d_w_in -= grad1;  }
    
    { auto grad1 = d_w_hr.getGrad(); grad1 *= lr; d_w_hr -= grad1;  }
    { auto grad1 = d_w_hz.getGrad(); grad1 *= lr; d_w_hz -= grad1;  }
    { auto grad1 = d_w_hn.getGrad(); grad1 *= lr; d_w_hn -= grad1;  }
  }

  void save(std::ostream& out) const override
  {
    d_w_ir.save(out);
    d_w_iz.save(out);
    d_w_in.save(out);
    
    d_w_hr.save(out);
    d_w_hz.save(out);
    d_w_hn.save(out);
  }
  void load(std::istream& in) override
  {
    d_w_ir.load(in);
    d_w_iz.load(in);
    d_w_in.load(in);
    
    d_w_hr.load(in);
    d_w_hz.load(in);
    d_w_hn.load(in);

  }
  void randomize() // "Xavier initialization"  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  {
    d_w_ir.randomize(1.0/sqrt(HIDDEN));
    d_w_iz.randomize(1.0/sqrt(HIDDEN));
    d_w_in.randomize(1.0/sqrt(HIDDEN));

    d_w_hr.randomize(1.0/sqrt(HIDDEN));
    d_w_hz.randomize(1.0/sqrt(HIDDEN));
    d_w_hn.randomize(1.0/sqrt(HIDDEN));
    d_prevh.zero();

    d_w_ir.needsGrad();
    d_w_iz.needsGrad();
    d_w_in.needsGrad();
    
    d_w_hr.needsGrad();
    d_w_hz.needsGrad();
    d_w_hn.needsGrad();
  }

  unsigned int size() const override
  {
    return d_w_ir.size() + d_w_iz.size() + d_w_in.size() +
           d_w_hr.size() + d_w_hz.size() + d_w_hn.size();
  }
  
  void zeroGrad() override
  {
    d_w_ir.zeroGrad();
    d_w_iz.zeroGrad();
    d_w_in.zeroGrad();

    d_w_hr.zeroGrad();
    d_w_hz.zeroGrad();
    d_w_hn.zeroGrad();
  }

  void addGrad(const GRULayer& rhs)
  {
    d_w_ir.addGrad(rhs.d_w_ir.getGrad());
    d_w_iz.addGrad(rhs.d_w_iz.getGrad());
    d_w_in.addGrad(rhs.d_w_in.getGrad());

    d_w_hr.addGrad(rhs.d_w_hr.getGrad());
    d_w_hz.addGrad(rhs.d_w_hz.getGrad());
    d_w_hn.addGrad(rhs.d_w_hn.getGrad());
    
  }

  void setGrad(const GRULayer& rhs, float divisor)
  {
    d_w_ir.setGrad(rhs.d_w_ir.getGrad()/divisor);
    d_w_iz.setGrad(rhs.d_w_iz.getGrad()/divisor);
    d_w_in.setGrad(rhs.d_w_in.getGrad()/divisor);

    d_w_hr.setGrad(rhs.d_w_hr.getGrad()/divisor);
    d_w_hz.setGrad(rhs.d_w_hz.getGrad()/divisor);
    d_w_hn.setGrad(rhs.d_w_hn.getGrad()/divisor);
  }

  template<typename W>
  void makeProj(const W& w)
  {
    d_w_ir_proj = ::makeProj(d_w_ir, w);
    d_w_iz_proj = ::makeProj(d_w_iz, w);
    d_w_in_proj = ::makeProj(d_w_in, w);

    d_w_hr_proj = ::makeProj(d_w_hr, w);
    d_w_hz_proj = ::makeProj(d_w_hz, w);
    d_w_hn_proj = ::makeProj(d_w_hn, w);

  }
  template<typename W>
  void projForward(W& w) const
  {
    ::projForward(d_w_ir_proj, d_w_ir, w);
    ::projForward(d_w_iz_proj, d_w_iz, w);
    ::projForward(d_w_in_proj, d_w_in, w);

    ::projForward(d_w_hr_proj, d_w_hr, w);
    ::projForward(d_w_hz_proj, d_w_hz, w);
    ::projForward(d_w_hn_proj, d_w_hn, w);
    
  }

  template<typename W>
  void projBackGrad(const W& w)
  {
    ::projBackGrad(d_w_ir_proj, w, d_w_ir);
    ::projBackGrad(d_w_iz_proj, w, d_w_iz);
    ::projBackGrad(d_w_in_proj, w, d_w_in);

    ::projBackGrad(d_w_hr_proj, w, d_w_hr);
    ::projBackGrad(d_w_hz_proj, w, d_w_hz);
    ::projBackGrad(d_w_hn_proj, w, d_w_hn);
  }

  
};

