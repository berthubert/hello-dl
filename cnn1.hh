#pragma once
#include "tracked.hh"
#include "layers.hh"

/*
Gratefully copied from 'mnist.cpp' in the PyTorch example repository
https://github.com/pytorch/examples/blob/main/cpp/mnist/mnist.cpp

This model takes MNIST 28*28 input and:

  * normalizes to "0.1307, 03081", torch::data::transforms::Normalize<>(0.1307, 0.3081)

  * applies a 5*5 kernel convolution `conv1`, configured to emit 10 layers, 24*24
  * does max_pool2d on these, which takes non-overlapping 2*2 rectangles 
    and emits max value per rectangle. Delivers 12*12 values for each layer 
  * ReLu
  * does another 5x5 convolution `conv2` on the 10 layers, turning them into 20 layers of 8*8
  * randomly *zeroes* half of the 20 layers `conv2_drop` - no state, Bernoulli 
    STILL MISSING!
  * another max_pool2d, 4*4*20 layers
  * ReLu
  * flatten to 320 values
  * linear combination 320x50 (fc1)
  * ReLU
  * zero out half of values randomly during training (STILL MISSING)
  * another linear combination, 50x10 (fc2)
  * log_softmax on the 10 values
  * the 10 outputs are probabilities per digit
  * highest probability is chosen
 */


struct CNNModel {
  NNArray<float, 28, 28> img;

  int label;
  NNArray<float, 10, 1> scores;
  NNArray<float, 1, 10> expected;

  TrackedFloat loss;

  struct State
  {
    //      R   C   K IN  OUTLAYERS
    Conv2d<28, 28, 5, 1,  10> c1; // -> 24*24 -> max2d -> 12*12
    Conv2d<12, 12, 5, 10, 20> c2; // -> 8*8 -> max2d -> 4*4

    //      IN OUT
    Linear<320, 50> fc1;
    Linear<50, 10> fc2;

    void learn(float lr)
    {
      c1.learn(lr);
      c2.learn(lr);
      fc1.learn(lr);
      fc2.learn(lr);
    }

    void save(std::ostream& out) const
    {
      c1.save(out);      c2.save(out);       fc1.save(out);       fc2.save(out);
    }
    void load(std::istream& in)
    {
      c1.load(in);      c2.load(in);       fc1.load(in);       fc2.load(in);
    }
  };
  
  void init(State& s)
  {
    img.zero();
                      
    auto step1 = s.c1.forward(img);
    
    std::array<NNArray<float, 12,12>, 10> step2;
    unsigned ctr=0;
    for(auto& p : step2)
      p = step1[ctr++].Max2d<2>().applyFunc(ReluFunc());

    std::array<NNArray<float, 8,8>, 20> step3 = s.c2.forward(step2);
    std::array<NNArray<float, 4,4>, 20> step4;

    ctr=0;
    for(auto& p : step4) {
      p = step3[ctr++].Max2d<2>().applyFunc(ReluFunc());
    }

    NNArray<float, 4*4*20, 1> flat = flatten(step4);
    auto output = s.fc1.forward(flat);
    auto output2 = output.applyFunc(ReluFunc());
    auto output3 = s.fc2.forward(output2);

    scores = output3.logSoftMax();
    expected.zero();
    loss = TrackedFloat(0) - (expected*scores)(0,0);
  }
};
