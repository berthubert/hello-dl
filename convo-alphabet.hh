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
  
  void init(State& s, bool production=false)
  {
    using ActFunc = GeluFunc;

    img.zero();
    img.d_imp->d_nograd=true;
    
    auto step1 = s.c1.forward(img);    // -> 26x26, 32 layers
    auto step2 = Max2dfw(step1, 2);    // -> 13x13
    auto step3 = s.c2.forward(step2);  // -> 11x11, 64 layers
    auto step4 = Max2dfw(step3, 2);    // -> 6x6 (padding)
    auto step5 = s.c3.forward(step4);  // -> 4x4, 128 layers
    auto step6 = Max2dfw(step5, 2);    // -> 2x2
    auto flat = makeFlatten(step6);    // -> 512x1
    auto output = s.fc1.forward(flat); // -> 64
    auto output1 = production ? output : output.makeDropout(0.5); 
    auto output2 = makeFunction<ActFunc>(output1);
    auto output3 = makeFunction<ActFunc>(s.fc2.forward(output2)); // -> 128
    auto output4 = makeFunction<ActFunc>(s.fc3.forward(output3)); // -> 26
    scores = makeLogSoftMax(output4);
    modelloss = -(expected*scores).sum();

    Tensor<float> fact(1,1);
    fact(0,0) = 0.02;
    weightsloss = fact*(s.c1.SquaredWeightsSum() +  s.c2.SquaredWeightsSum() +  s.c3.SquaredWeightsSum() +
                        s.fc1.SquaredWeightsSum() + s.fc1.SquaredWeightsSum() + s.fc1.SquaredWeightsSum());

    loss = modelloss; // + weightsloss;
  }
};
