#pragma once
#include "tensor-layers.hh"
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
struct GRULayer : TensorLayer<T>
{
  Tensor<T> d_w_ir{HIDDEN, IN}; // reset
  Tensor<T> d_w_iz{HIDDEN, IN}; // update
  Tensor<T> d_w_in{HIDDEN, IN}; // new

  Tensor<T> d_w_hr{HIDDEN, HIDDEN};  // hidden reset
  Tensor<T> d_w_hz{HIDDEN, HIDDEN}; // hidden update
  Tensor<T> d_w_hn{HIDDEN, HIDDEN}; // hidden "new"

  Tensor<T> d_origprevh{HIDDEN, 1};
  Tensor<T> d_prevh{HIDDEN, 1};

  GRULayer()
  {
    this->d_params={
        {&d_w_ir, "w_ir"},         {&d_w_iz, "w_iz"},         {&d_w_in, "w_in"},
        {&d_w_hr, "w_hr"},         {&d_w_hz, "w_hz"},         {&d_w_hn, "w_hn"}};
    randomize();
    Tensor one(HIDDEN, HIDDEN);
    one.identity(1.0);
    d_prevh = one*d_origprevh;
  }

  // https://blog.floydhub.com/gru-with-pytorch/
  //  https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18
  // these appear to be slightly different
  auto forward(const Tensor<T>& xt)
  {
    auto r_t = makeFunction<SigmoidFunc>(d_w_ir * xt + d_w_hr * d_prevh); // reset gate
    auto z_t = makeFunction<SigmoidFunc>(d_w_iz * xt + d_w_hz * d_prevh);
    // z_t dimensions: rows from d_w_iz, columns from xt -> HIDDEN,IN
    auto n_t = makeFunction<TanhFunc>(d_w_in * xt + r_t.dot(d_w_hn *d_prevh));

    Tensor<T> one(HIDDEN, 1);  // XXX this is a SUPER wart
    // the problem is we have no support for 1 - Tensor() kind of operations, so we need to make an appropriately sized 'one'
    for(unsigned int r=0 ; r < one.getRows(); ++r)
      for(unsigned int c=0 ; c < one.getCols(); ++c)
        one(r,c)=1;
    
    auto h_t = (one - z_t).dot(n_t) + z_t.dot(d_prevh);
    d_prevh = h_t; // "this is where the magic happens"
    return h_t;
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
  }
};
