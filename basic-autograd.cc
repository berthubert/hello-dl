#include "tensor2.hh"
#include <iostream>
using namespace std;

int main()
{
#if 0
  Tensor x(2.0f);
  Tensor z(0.0f);
  Tensor a(1.0f);
  Tensor y = x * (z + a);
  y(0,0);
  y.backward();
  
#else 
  Tensor x(2.0f);
  Tensor z(0.0f);
  Tensor y = Tensor(3.0f)*x*x*x + Tensor(4.0f)*x + Tensor(1.0f) + x*z;
  y(0,0);
  y.backward();  
  cout << "y = "<< y << endl; // 3*8 + 4*2 + 1 = 33
  


  cout << "dy/dx = " << x.getGrad() << endl; // 9*x^2 + 4 = 40
  cout << "dy/dz = " << z.getGrad() << endl; // 2
  #endif
}
