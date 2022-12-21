#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

extern std::ofstream g_tree;

constexpr bool doLog{false};

struct ReluFunc
{
  static float func(const float& in)
  {
    if(in < 0)
      return 0;
    else
      return in;
  }
  static float deriv(const float& in)
  {
    if(in < 0)
      return 0;
    else
      return 1;
  }
#if 0
  static Eigen::MatrixXf func(const Eigen::MatrixXf& in)
  {
    return in.unaryExpr([](const float& v) { return func(v);});
  }
  static Eigen::MatrixXf deriv(const Eigen::MatrixXf& in)
  {
    return in.unaryExpr([](const float& v) {return deriv(v);});
  }
#endif
  static std::string getName() { return "relu"; }
};

struct SigmoidFunc
{
  static float func(const float& in)
  {
    return 1.0/ (1.0 +exp(-in));
  }
  static float deriv(const float& in)
  {
    float sigma = 1.0/ (1.0 +exp(-in));
    return sigma*(1-sigma);
  }

  #if 0
  static Eigen::MatrixXf func(const Eigen::MatrixXf& in)
  {
    return in.unaryExpr([](const float& v) -> float {
      return 1.0/ (1.0 +exp(-v));
    });
  }
  static Eigen::MatrixXf deriv(const Eigen::MatrixXf& in)
  {
    return in.unaryExpr([](const float& v) -> float {
      float sigma = 1.0/ (1.0 +exp(-v));
      return sigma*(1-sigma);
    });
  }
  #endif
  static std::string getName() { return "sigmoid"; }
};


struct SinFunc
{
  static Eigen::MatrixXf func(const Eigen::MatrixXf& in)
  {
    return in.unaryExpr([](const float& v) -> float {
      return sin(v);
    });
  }
  static Eigen::MatrixXf deriv(const Eigen::MatrixXf& in)
  {
    return in.unaryExpr([](const float& v) -> float {
      return cos(v);
    });
  }
  static std::string getName() { return "sin"; }
};

struct IDFunc
{
  static float func(const float& in)
  {
    return in;
  }
  static float deriv([[maybe_unused]]const float& in)
  {
    return 1;
  }

  static Eigen::MatrixXf func(const Eigen::MatrixXf& in)
  {
    return in;
  }
  static Eigen::MatrixXf deriv(const Eigen::MatrixXf& in)
  {
    return in.unaryExpr([](const float& v) { return deriv(v);});
  }
  static std::string getName() { return "ID"; }
};


template<typename T>
struct TrackedNumberImp
{
  static void setZero(float& v)
  {
    v=0;
  }
  
  static void setZero(Eigen::MatrixXf& v)
  {
    v.setZero();
  }
  
  TrackedNumberImp(){}
  explicit TrackedNumberImp(T v) : d_val(v), d_mode(Modes::Parameter)
  {
    d_grad = v; // get the dimensions right for matrix
    setZero(d_grad);
  }
  
  T getVal() const
  {
    if(d_mode == Modes::Parameter)
      return d_val;
    else if(d_mode == Modes::Addition) 
      return d_val=(d_lhs->getVal() + d_rhs->getVal());
    else if(d_mode == Modes::Mult) 
      return d_val=(d_lhs->getVal() * d_rhs->getVal());
    else if(d_mode == Modes::Func) {
      d_val = d_func(d_lhs->getVal());
      return d_val;     
    }
    else
      abort();
  }

  void zeroGrad()
  {
    setZero(d_grad);
    if(d_lhs)
      d_lhs->zeroGrad();
    if(d_rhs)
      d_rhs->zeroGrad();
  }
  
  T getGrad() 
  {
    return d_grad;
  }

  float transpose(float v)
  {
    return v;
  }

  Eigen::MatrixXf transpose(const Eigen::MatrixXf& v)
  {
    return v.transpose();
  }
  void backward(T mult) 
  {
    if(d_mode == Modes::Parameter) {
      if(doLog) std::cout<<"I'm a param called '"<<getName()<<"' with value "<<d_val<<", nothing to backward further"<<std::endl;
      d_grad += mult;
      if(doLog) std::cout<<"   receiving a mult of "<< (mult) <<std::endl;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<getName()<<"="<<d_val<<"\\ng="<<d_grad<<"\"]\n";;
      return;
    }
    else if(d_mode == Modes::Addition) {
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n+\"]\n";;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      if(doLog) std::cout<<"Addition going left, delivering "<< (mult)<<std::endl;
      d_lhs->backward(mult);
      if(doLog) std::cout<<"Addition going right, delivering "<< (mult) <<std::endl;
      d_rhs->backward(mult);
    }
    else if(d_mode == Modes::Mult) {
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n*\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      if(doLog) std::cout<<"Mult lhs grad: \n"<<d_lhs->d_grad<<std::endl;
      T tposed = transpose(d_rhs->d_val);

      if(doLog) std::cout<<"Our grad:\n";
      if(doLog) std::cout<<d_grad<<std::endl;

      if(doLog) std::cout<<"Mult rhs grad: \n"<<d_rhs->d_grad<<std::endl;
      
      T tposed2 = transpose(d_lhs->d_val);
      
      if(doLog) std::cout<<"Going to left, delivering "<< (mult * tposed )<<std::endl;
      d_lhs->backward(mult * tposed);
      if(doLog) std::cout<<"Going right, delivering " << (tposed2 * mult ) <<std::endl;
      d_rhs->backward(tposed2 * mult);
    }
    else if(d_mode == Modes::Func) {
      if(doLog) std::cout<<"Function... "<<std::endl;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<"func\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";

      d_lhs->backward(mult*d_deriv(d_lhs->d_val));
    }
    else
      abort();
  }
  enum class Modes
  {
    Parameter,
    Addition,
    Mult,
    Func
  };
  std::shared_ptr<TrackedNumberImp> d_lhs, d_rhs;  
  mutable T d_val; // 4
  T d_grad; // 4
  typedef float(*func_t)(const float&);
  func_t d_func, d_deriv;
  //  std::string d_funcname;

  Modes d_mode;


  //  std::string d_name;
  std::string getName() { return "none"; }
};

template<typename T>
struct TrackedNumber
{
  TrackedNumber(){}
  TrackedNumber(T val, [[maybe_unused]] const std::string& name="")
  {
    impl = std::make_shared<TrackedNumberImp<T>>(val);
    //    impl->d_name = name;
  }
  T getVal() const
  {
    return impl->getVal();
  }
  T getGrad() const
  {
    return impl->getGrad();
  }

  void zeroGrad()
  {
    impl->zeroGrad();
  }
  TrackedNumber& operator=(T val)
  {
    if(!impl)
      impl = std::make_shared<TrackedNumberImp<T>>(val);
    else {
      if(impl->d_mode != TrackedNumberImp<T>::Modes::Parameter) {
        std::cerr<<"Trying to assign a number to something not a number"<<std::endl;
        abort();
      }
      impl->d_val = val;
    }
    return *this;
  }
  void setOne(float& mul)
  {
    mul=1;
  }

  void setOne(Eigen::MatrixXf& mul)
  {
    mul.setConstant(1);
  }
  void backward()
  {
    T mul;
    setOne(mul);
    impl->backward(mul);
  }
  std::shared_ptr<TrackedNumberImp<T>> impl;
};

template<typename T>
TrackedNumber<T> operator+(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Addition;
  ret.impl->d_lhs = lhs.impl;
  ret.impl->d_rhs = rhs.impl;
  //  ret.impl->d_grad = lhs.getVal(); // dimensions
  TrackedNumberImp<T>::setZero(ret.impl->d_grad);
  return ret;
}

template<typename T>
TrackedNumber<T> operator-(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  return lhs + TrackedNumber<T>(-1)*rhs;
}


template<typename T>
TrackedNumber<T> operator*(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Mult;
  assert(lhs.impl != 0);
  ret.impl->d_lhs = lhs.impl;
  assert(rhs.impl != 0);
  ret.impl->d_rhs = rhs.impl;
  //  ret.impl->d_grad = lhs.getVal() * rhs.getVal(); // get dimensions right
  // ret.impl->d_grad.setConstant(1);  // XXX maybe should come back
  return ret;
}

template<typename T, typename F>
TrackedNumber<T> doFunc(const TrackedNumber<T>& lhs, [[maybe_unused]] const F& f)
{
  
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Func;
  ret.impl->d_func = F::func;
  ret.impl->d_deriv = F::deriv;
  ret.impl->d_lhs = lhs.impl;
  
  //  ret.impl->d_grad = lhs.getVal(); // get dimensions right
  //  ret.impl->d_grad.setConstant(1); // no idea
  //  ret.impl->d_funcname = F::getName();
  return ret;
}



typedef TrackedNumber<float> TrackedFloat;
