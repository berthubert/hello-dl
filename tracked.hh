#pragma once
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <unordered_set>
extern std::ofstream g_tree;

constexpr bool doLog{false};

struct SquareFunc
{
  static float func(const float& v)
  {
    return v*v;
  }
  static float deriv(const float& v)
  {
    return 2*v;
  }
};

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

  static std::string getName() { return "sigmoid"; }
};

struct ExpFunc
{
  static float func(const float& in)
  {
    if(in<80)
      return exp(in);
    else
      return exp(80);
  }
  static float deriv(const float& in)
  {
    if(in<87)
      return exp(in); // easy
    else
      return 0;
  }

  static std::string getName() { return "exp"; }
};

struct LogFunc
{
  static float func(const float& in)
  {
    if(in == 0.0)
      return -80;
    return log(in);
  }
  static float deriv(const float& in)
  {
    if(in==0.0)
      return 80;
    return 1/in;
  }

  static std::string getName() { return "log"; }
};

static uint64_t s_count;

template<typename T>
struct TrackedNumberImp
{
  static void setZero(float& v)
  {
    v=0;
  }
  
  TrackedNumberImp(const TrackedNumberImp&) = delete;
  TrackedNumberImp& operator=(const TrackedNumberImp&) = delete;
  TrackedNumberImp()
  {
    s_count++;
  }
  explicit TrackedNumberImp(T v) : d_val(v), d_mode(Modes::Parameter)
  {
    d_grad = v; // get the dimensions right for matrix
    setZero(d_grad);
    s_count++;
  }

  ~TrackedNumberImp()
  {
    --s_count;
  }

  static uint64_t getCount()
  {
    return s_count;
  }
  
  T getVal() const
  {
    if(d_mode == Modes::Parameter || d_haveval)
      return d_val;
    else if(d_mode == Modes::Addition) {
      d_val=(d_lhs->getVal() + d_rhs->getVal());
    }
    else if(d_mode == Modes::Mult) {
      d_val=(d_lhs->getVal() * d_rhs->getVal());
    }
    else if(d_mode == Modes::Div)
      d_val=(d_lhs->getVal() / d_rhs->getVal());
    else if(d_mode == Modes::Func) {
      d_val = d_func(d_lhs->getVal());
    }
    else if(d_mode == Modes::Max) {
      T l = d_lhs->getVal(), r=d_rhs->getVal();
      d_val = std::max(l, r);
    }
    else
      abort();
    d_haveval=true;
    return d_val;
  }

  void zeroGrad()
  {
    setZero(d_grad);
    d_haveval=false;
  }
  
  T getGrad() 
  {
    return d_grad;
  }

  float transpose(float v)
  {
    return v;
  }

  void build_topo(std::unordered_set<TrackedNumberImp<T>*>& visited, std::vector<TrackedNumberImp<T>*>& topo)
  {
    if(visited.count(this))
      return;
    visited.insert(this);
    
    if(d_lhs) {
      d_lhs->build_topo(visited, topo);
    }
    if(d_rhs) {
      d_rhs->build_topo(visited, topo);
    }
    topo.push_back(this);

  }

  void doGrad()
  {
    if(d_mode == Modes::Parameter) {
      if(doLog) std::cout<<"I'm a param called '"<<getName()<<"' with value "<<d_val<<", nothing to backward further"<<std::endl;

      if(doLog) std::cout<<"   receiving a d_grad of "<< (d_grad) <<std::endl;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<getName()<<"="<<d_val<<"\\ng="<<d_grad<<"\"]\n";;
      return;
    }
    else if(d_mode == Modes::Addition) {
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n+\"]\n";;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      if(doLog) std::cout<<"Addition going left, delivering "<< (d_grad)<<std::endl;

      d_lhs->d_grad += d_grad;
      if(doLog) std::cout<<"Addition going right, delivering "<< (d_grad) <<std::endl;

      d_rhs->d_grad+=d_grad;
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
      
      if(doLog) std::cout<<"Going to left, delivering "<< (d_grad * tposed )<<std::endl;

      d_lhs->d_grad+=(d_grad * tposed);
      if(doLog) std::cout<<"Going right, delivering " << (tposed2 * d_grad ) <<std::endl;

      d_rhs->d_grad+=(tposed2 * d_grad);
    }
    else if(d_mode == Modes::Div) {
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n/\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      if(doLog) std::cout<<"Div lhs grad: \n"<<d_lhs->d_grad<<std::endl;
      T tposed = transpose(d_rhs->d_val);

      if(doLog) std::cout<<"Our grad:\n";
      if(doLog) std::cout<<d_grad<<std::endl;

      if(doLog) std::cout<<"Div rhs grad: \n"<<d_rhs->d_grad<<std::endl;
      
      T tposed2 = transpose(d_lhs->d_val);
      
      if(doLog) std::cout<<"Going to left, delivering "<< ( d_grad/tposed )<<std::endl;

      d_lhs->d_grad+=(d_grad/tposed);
      if(doLog) std::cout<<"Going right, delivering " << (-d_grad*tposed2/(tposed*tposed) ) <<std::endl;

      d_rhs->d_grad+=(-d_grad*tposed2/(tposed*tposed));
    }
    else if(d_mode == Modes::Func) {
      if(doLog) std::cout<<"Function... "<<std::endl;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<"func\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";

      d_lhs->d_grad+=(d_grad*d_deriv(d_lhs->d_val));
    }
    else if(d_mode == Modes::Max) {
      if(doLog) std::cout<<"Max... "<<std::endl;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<"max\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";

      if(d_lhs->d_val < d_rhs->d_val)
        d_rhs->d_grad+= d_grad;
      else
        d_lhs->d_grad+= d_grad;
    }

    else
      abort();
  }
  
  // inspiration: https://github.com/flashlight/flashlight/blob/main/flashlight/fl/autograd/Variable.cpp
  // more inspiration: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
  enum class Modes : uint8_t
  {
    Unassigned = 0,
    Parameter=1,
    Addition=2,
    Mult=3,
    Div=4,
    Func=5,
    Max=6
  };
  std::shared_ptr<TrackedNumberImp> d_lhs, d_rhs;  
  mutable T d_val; // 4
  T d_grad{0}; // 4
  typedef float(*func_t)(const float&);
  func_t d_func, d_deriv;
  //  std::string d_funcname;

  Modes d_mode{Modes::Unassigned};
  mutable bool d_haveval{false};

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
    auto topo=getTopo();
    zeroGrad(topo);
  }

  void zeroGrad(std::vector<TrackedNumberImp<T>* > topo)
  {
    for(auto iter = topo.rbegin(); iter != topo.rend(); ++iter) {
      (*iter)->zeroGrad();
    }
  }

  
  TrackedNumber& operator=(T val)
  {
    if(!impl)
      impl = std::make_shared<TrackedNumberImp<T>>(val);
    else {
      if(impl->d_mode != TrackedNumberImp<T>::Modes::Parameter) {
        std::cerr<<"Trying to assign a number to something not a number, it is a "<<(int)impl->d_mode<<std::endl;
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

  std::vector<TrackedNumberImp<T>* > getTopo()
  {
    std::vector<TrackedNumberImp<T>* > topo;
    std::unordered_set<TrackedNumberImp<T>* > visited;
    impl->build_topo(visited, topo);
    return topo;
  }
  
  std::vector<TrackedNumberImp<T>* > backward()
  {
    //    std::cout << "Starting topo" << std::endl;
    auto topo=getTopo();
    backward(topo);
    return topo;
  }

  void backward(std::vector<TrackedNumberImp<T>* >& topo)
  {
    impl->d_grad = 1.0;
    //    std::cout<<"Have "<<topo.size()<<" entries to visit\n";
    for(auto iter = topo.rbegin(); iter != topo.rend(); ++iter) {
      (*iter)->doGrad();
    }
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
  return lhs + (TrackedNumber<T>(-1)*rhs);
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

template<typename T>
TrackedNumber<T> operator/(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Div;
  assert(lhs.impl != 0);
  ret.impl->d_lhs = lhs.impl;
  assert(rhs.impl != 0);
  ret.impl->d_rhs = rhs.impl;
  return ret;
}

template<typename T>
TrackedNumber<T> makeMax(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Max;
  assert(lhs.impl != 0);
  ret.impl->d_lhs = lhs.impl;
  assert(rhs.impl != 0);
  ret.impl->d_rhs = rhs.impl;
  return ret;
}

template<typename T, typename F>
TrackedNumber<T> makeFunc(const TrackedNumber<T>& lhs, [[maybe_unused]] const F& f)
{
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Func;
  ret.impl->d_func = F::func;
  ret.impl->d_deriv = F::deriv;
  ret.impl->d_lhs = lhs.impl;
  
  return ret;
}

typedef TrackedNumber<float> TrackedFloat;
