#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <Eigen/Dense>
#include <optional>

using namespace Eigen;
using namespace std;


static ofstream tree("tree.part");

template<typename T>
struct TrackedNumberImp
{
  TrackedNumberImp(){}
  explicit TrackedNumberImp(T v) : d_val(v), d_mode(Modes::Parameter)
  {
    d_grad = v; // get the dimensions right for matrix
    d_grad.setZero();
  }
  mutable T d_val;
  T d_grad;


  T getVal() const
  {
    if(d_mode == Modes::Parameter)
      return d_val;
    else if(d_mode == Modes::Addition) 
      return d_val=(d_lhs->getVal() + d_rhs->getVal());
    else if(d_mode == Modes::Mult) 
      return d_val=(d_lhs->getVal() * d_rhs->getVal());
    else
      abort();
  }

  T getGrad() 
  {
    return d_grad;
  }

  void backward(T mult) 
  {
    if(d_mode == Modes::Parameter) {
      cout<<"I'm a digit called '"<<d_name<<"', nothing to backward further"<<endl;
      d_grad += mult;
      cout<<"   receiving a mult of "<< (mult) <<endl;
      tree<<'"'<<(void*)this<< "\" [label=\""<<d_name<<"="<<d_val<<"\\ng="<<d_grad<<"\"]\n";;
      return;
    }
    else if(d_mode == Modes::Addition) {
      tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n+\"]\n";;
      tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      cout<<"Addition going left, delivering "<< (mult)<<endl;
      d_lhs->backward(mult);
      cout<<"Addition going right, delivering "<< (mult) <<endl;
      d_rhs->backward(mult);
    }
    else if(d_mode == Modes::Mult) {
      tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n*\"]\n";;
      tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      cout<<"Mult lhs grad: \n"<<d_lhs->d_grad<<endl;
      MatrixXf tposed = d_rhs->d_val.transpose(); 

      cout<<"Our grad:\n";
      cout<<d_grad<<endl;

      cout<<"Mult rhs grad: \n"<<d_rhs->d_grad<<endl;
      
      MatrixXf tposed2 = d_lhs->d_val.transpose();
      
      //d_rhs->d_grad.colwise() += tposed2.col(0);
      // AND DO GRAD!! XXX

      //      d_lhs->d_grad.rowwise() +=  d_rhs->getVal().transpose();// * d_grad;
      //      d_rhs->d_grad.rowwise() +=  d_lhs->getVal().transpose();// * d_grad;
      cout<<"Going to left, delivering "<< (mult * tposed )<<endl;
      d_lhs->backward(mult * tposed);
      cout<<"Going right, delivering " << (tposed2 * mult ) <<endl;
      d_rhs->backward(tposed2 * mult);
    }
    else
      abort();
  }
  enum class Modes
  {
    Parameter,
    Addition,
    Mult
  };

  
  Modes d_mode;
  shared_ptr<TrackedNumberImp> d_lhs, d_rhs;
  string d_name;
};

template<typename T>
struct TrackedNumber
{
  TrackedNumber(){}
  TrackedNumber(T val, const std::string& name="")
  {
    impl = make_shared<TrackedNumberImp<T>>(val);
    impl->d_name = name;
  }
  T getVal() const
  {
    return impl->getVal();
  }
  T getGrad() const
  {
    return impl->getGrad();
  }
  void backward()
  {
    Matrix<float, 1,1> mul;
    mul.setConstant(1);
    impl->backward(mul);
  }
  shared_ptr<TrackedNumberImp<T>> impl;
};

template<typename T>
TrackedNumber<T> operator+(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  TrackedNumber<T> ret;
  ret.impl = make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Addition;
  ret.impl->d_lhs = lhs.impl;
  ret.impl->d_rhs = rhs.impl;
  ret.impl->d_grad = lhs.getVal(); // dimensions
  ret.impl->d_grad.setZero(); 
  return ret;
}

template<typename T>
TrackedNumber<T> operator*(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  TrackedNumber<T> ret;
  ret.impl = make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Mult;
  ret.impl->d_lhs = lhs.impl;
  ret.impl->d_rhs = rhs.impl;
  ret.impl->d_grad = lhs.getVal() * rhs.getVal(); // get dimensions right
  ret.impl->d_grad.setConstant(1); 
  return ret;
}

typedef TrackedNumber<float> TrackedFloat;

int main()
{
  tree<<"digraph G { \n";
  cout<<"Start!"<<endl;

  cout<<"---- "<<endl;
  typedef TrackedNumber<MatrixXf> TrackedMatrix;

  if(1)
  {
    TrackedMatrix input(MatrixXf::Random(8,1), "input"), weights(MatrixXf::Random(2,8), "weights");
    Matrix<float, 1, 2> tmp;
    tmp<<1,2;
    TrackedMatrix final(tmp, "final");
    cout<<"Gradients should be all zeros, horizontal, two rows: "<<endl;
    cout<<weights.getGrad()<<endl;

    cout<<"Input, should be 1 column: "<<endl;
    cout<<input.getVal()<<endl;

    cout<<"Final, should be 1 2: "<<endl;
    cout<<final.getVal()<<endl;
    
    TrackedMatrix res = weights*input; 
    cout<<"Result should be two numbers"<<endl;
    cout<<res.getVal()<<endl;

    TrackedMatrix output = final * res;
    cout<<"Output, should be one number: "<<output.getVal()<<endl;
    cout<<"--- starting backward ---"<<endl;
    output.backward();
    cout<<"And the grad: "<<endl;
    cout<<weights.getGrad()<<endl;
  }

  if(0)
  {
    TrackedMatrix x(MatrixXf::Constant(1,1, 2), "x");
    TrackedMatrix y(MatrixXf::Constant(1,1, 3), "y");
    TrackedMatrix z(MatrixXf::Constant(1,1, 4), "z");
    TrackedMatrix res = y * (x + x + x + x ) + z; //  y*(4*x)
    // 
    cout<<"Result: "<<res.getVal()<<endl;
    res.backward();
    cout<<x.getGrad()<<endl;  // 4*y = 12
    cout<<y.getGrad()<<endl;  // 4*x = 8
    cout<<z.getGrad()<<endl;  // 1
  }

  if(0)
  {
    TrackedMatrix x(MatrixXf::Constant(1,1, 2), "x");
    TrackedMatrix y(MatrixXf::Constant(1,1, 3), "y");
    TrackedMatrix z(MatrixXf::Constant(1,1, 4.5), "z");
    TrackedMatrix w1(MatrixXf::Constant(1,1, 1.0), "w");
    TrackedMatrix w2(MatrixXf::Constant(1,1, 2.0), "w");
    TrackedMatrix res = (w1+w2) * (x * x * x * y + z * z); 
    // 
    cout<<"Result: "<<res.getVal()<<endl;
    res.backward();
    cout<<x.getGrad()<<endl;  // 3*x^2*y = 36
    cout<<y.getGrad()<<endl;  // x^3 = 8
    cout<<z.getGrad()<<endl;  // 2*z = 9
  }

  tree<<"}"<<endl;
}

