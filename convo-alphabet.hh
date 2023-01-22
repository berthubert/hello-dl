#include "tensor-layers.hh"

struct ConvoAlphabetModel {
  Tensor<float> img{28,28};
  Tensor<float> scores{26, 1};
  Tensor<float> expected{1,26};
  Tensor<float> modelloss{1,1};
  Tensor<float> weightsloss{1,1};
  Tensor<float> loss{1,1};

  struct State : public ModelState<float>
  {
    //           r_in c   k c_i  c_out
    Conv2d<float, 28, 28, 3, 1,  32> c1; // -> 26*26 -> max2d -> 13*13
    Conv2d<float, 13, 13, 3, 32, 64> c2; // -> -> 11*11 -> max2d -> 6*6 //padding
    Conv2d<float, 6,   6, 3, 64, 128> c3; // -> 4*4 -> max2d -> 2*2
    // flattened to 512 (128*2*2)
           //      IN OUT
    Linear<float, 512, 64> fc1;  
    Linear<float, 64, 128> fc2;
    Linear<float, 128, 26> fc3; 

    State()
    {
      this->d_members = {{&c1, "c1"}, {&c2, "c2"}, {&c3, "c3"}, {&fc1, "fc1"}, {&fc2, "fc2"}, {&fc3, "fc3"}};
    }
  };
  
  void init(State& s)
  {
    img.zero();
    img.d_imp->d_nograd=true;
    auto step1 = s.c1.forward(img);

    using ActFunc = GeluFunc;
    
    std::array<Tensor<float>, 32> step2; // 13x13
    unsigned ctr=0;
    for(auto& p : step2)
      p = makeFunction<ActFunc>(step1[ctr++].makeMax2d(2));

    std::array<Tensor<float>, 64> step3 = s.c2.forward(step2);  // 11x11
    std::array<Tensor<float>, 64> step4;                   // 6x6

    ctr=0;
    for(auto& p : step4) {
      p = makeFunction<ActFunc>(step3[ctr++].makeMax2d(2));
    }

    std::array<Tensor<float>, 128> step5 = s.c3.forward(step4); // 4x4
    std::array<Tensor<float>, 128> step6;                  // 2x2

    ctr=0;
    for(auto& p : step6) {
      p = makeFunction<ActFunc>(step5[ctr++].makeMax2d(2));
    }
    
    Tensor<float> flat = makeFlatten(step6); // 2*2*128 * 1
    auto output = s.fc1.forward(flat);
    auto output2 = makeFunction<ActFunc>(output);
    auto output3 = makeFunction<ActFunc>(s.fc2.forward(output2));
    auto output4 = makeFunction<ActFunc>(s.fc3.forward(output3));
    scores = makeLogSoftMax(output4);
    modelloss = -(expected*scores).sum();

    Tensor<float> fact(1,1);
    fact(0,0) = 0.02;
    weightsloss = fact*(s.c1.SquaredWeightsSum() +  s.c2.SquaredWeightsSum() +  s.c3.SquaredWeightsSum() +
                        s.fc1.SquaredWeightsSum() + s.fc1.SquaredWeightsSum() + s.fc1.SquaredWeightsSum());

    loss = modelloss; // + weightsloss;
  }
};
