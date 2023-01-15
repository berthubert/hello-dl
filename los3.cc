#define EIGEN_USE_THREADS
#include <iostream>
#include <stdlib.h>
#include "tensor2.hh"
#include "misc.hh"

#include <Eigen/CXX11/Tensor>
using namespace std;

int main(int argc, char**argv)
{
  DTime dt;
  dt.start();
  float t=0;
  for(unsigned int n=0; n < 10000/64; ++n) {
    Eigen::Tensor<float, 3> input(64,28, 28);
    Eigen::Tensor<float, 2> kernel(3, 3);
    Eigen::Tensor<float, 3> output(64, 26, 26);
    input.setRandom();
    kernel.setRandom();
    
    Eigen::array<ptrdiff_t, 2> dims({1, 2});  // Specify second and third dimension for convolution.
    output = input.convolve(kernel, dims);
    t+=output(0,0,0);
  }
  cout<<"t: "<<t<<", "<<dt.lapUsec() << endl;
  t=0;
  dt.start();
  for(unsigned int n =0 ; n <10000; ++n) {
    Tensor i(28,28);
    Tensor k(3,3);
    Tensor b(1,1);
    i.randomize(1);
    k.randomize(1);
    b.randomize(1);
    Tensor c=i.makeConvo(3, k, b);

    t += c(0,0);
  }
  cout<<"t: "<<t<<", "<<dt.lapUsec() << endl;
  
}
