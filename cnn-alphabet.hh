#pragma once
#include "model.hh"
#include <sstream>
/*
Model from https://data-flair.training/blogs/handwritten-character-recognition-neural-network/
 */


template<typename T>
struct CNNAlphabetModel  {
  NNArray<T, 28, 28> img;

  NNArray<T, 26, 1> scores;
  NNArray<T, 1, 26> expected;

  TrackedNumber<T> loss;

  struct State : ModelState
  {
           //      R   C  K IN  OUTLAYERS
    Conv2d<T, 28, 28, 3, 1,  32> c1; // -> 26*26 -> max2d -> 13*13
    Conv2d<T, 13, 13, 3, 32, 64> c2; // -> -> 11*11 -> max2d -> 6*6 //padding
    Conv2d<T, 6,   6, 3, 64, 128> c3; // -> 4*4 -> max2d -> 2*2
    // flattened to 512
           //      IN OUT
    Linear<T, 512, 64> fc1;  
    Linear<T, 64, 128> fc2;
    Linear<T, 128, 26> fc3;

        // this has GRAVE issues with copying
    State() 
    {    // this has GRAVE issues with copying
      d_members={&c1, &c2, &c3, &fc1, &fc2, &fc3};
    }
    // this has GRAVE issues with copying
    State(const State& rhs) 
    {
      c1=rhs.c1;
      c2=rhs.c2;
      c3=rhs.c3;
      fc1=rhs.fc1;
      fc2=rhs.fc2;
      fc3=rhs.fc3;
      d_members={&c1, &c2, &c3, &fc1, &fc2, &fc3};
    }

    State& operator=(const State& rhs) = delete;
    
    void reset()
    {
      for(auto& m : d_members)
        m->reset();
    }
    
    void addGrad(const State& rhs)
    {
      c1.addGrad(rhs.c1);
      c2.addGrad(rhs.c2);
      c3.addGrad(rhs.c3);

      fc1.addGrad(rhs.fc1);
      fc2.addGrad(rhs.fc2);
      fc3.addGrad(rhs.fc3);
    }

    void setGrad(const State& rhs, float divisor)
    {
      c1.setGrad(rhs.c1, divisor);
      c2.setGrad(rhs.c2, divisor);
      c3.setGrad(rhs.c3, divisor);

      fc1.setGrad(rhs.fc1, divisor);
      fc2.setGrad(rhs.fc2, divisor);
      fc3.setGrad(rhs.fc3, divisor);
    }

    void momentum(const State& rhs, float mom, float dampening=0)
    {
      c1.momGrad(rhs.c1, mom, dampening);
      c2.momGrad(rhs.c2, mom, dampening);
      c3.momGrad(rhs.c3, mom, dampening);

      fc1.momGrad(rhs.fc1, mom, dampening);
      fc2.momGrad(rhs.fc2, mom, dampening);
      fc3.momGrad(rhs.fc3, mom, dampening);
    }

    
    template<typename W>
    void makeProj(const W& w)
    {
      fc1.makeProj(w);
      fc2.makeProj(w);
      fc3.makeProj(w);
      c1.makeProj(w);
      c2.makeProj(w);
    }

    template<typename W>
    void projForward(W& w) const
    {
      fc1.projForward(w);
      fc2.projForward(w);
      fc3.projForward(w);
      c1.projForward(w);
      c2.projForward(w);
    }
    template<typename W>
    void projBackGrad(const W& w) 
    {
      fc1.projBackGrad(w);
      fc2.projBackGrad(w);
      fc3.projBackGrad(w);
      c1.projBackGrad(w);
      c2.projBackGrad(w);
    }

  };
  
  void init(State& s)
  {
    img.zero();
    img.setVariable();
                      
    auto step1 = s.c1.forward(img);
    
    std::array<NNArray<T, 13,13>, 32> step2;
    unsigned ctr=0;
    for(auto& p : step2)
      p = step1[ctr++]. template Max2d<2>().applyFunc(ReluFunc());

    std::array<NNArray<T, 11,11>, 64> step3 = s.c2.forward(step2);
    std::array<NNArray<T, 6,6>, 64> step4;

    ctr=0;
    for(auto& p : step4) {
      p = step3[ctr++]. template Max2d<2>().applyFunc(ReluFunc());
    }

    std::array<NNArray<T, 4,4>, 128> step5 = s.c3.forward(step4);
    std::array<NNArray<T, 2,2>, 128> step6;

    ctr=0;
    for(auto& p : step6) {
      p = step5[ctr++]. template Max2d<2>().applyFunc(ReluFunc());
    }
    
    NNArray<T, 2*2*128, 1> flat = flatten(step6);
    auto output = s.fc1.forward(flat);
    auto output2 = output.applyFunc(ReluFunc());
    auto output3 = s.fc2.forward(output2).applyFunc(ReluFunc());
    auto output4 = s.fc3.forward(output3).applyFunc(ReluFunc());

    scores = output4.logSoftMax();
    expected.zero();
    loss = TrackedNumber<T>(0.0) - (expected*scores)(0,0);
  }

};
