#pragma once
#include <vector>
#include <memory>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <unordered_set>
#include "fvector.hh"
#include "trackedfuncs.hh"
extern std::ofstream g_tree;

constexpr bool doLog{false};

inline float maxFunc(float a, float b)
{
  return std::max(a,b);
}

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
    d_grad = 0.0; 
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
      d_val = getFunc<T>(g_fss[d_findex])(d_lhs->getVal());
    }
    else if(d_mode == Modes::Max) {
      T l = d_lhs->getVal(), r=d_rhs->getVal();
      d_val = maxFunc(l, r);
    }
    else
      abort();
    d_haveval=true;
    return d_val;
  }

  void zeroGrad()
  {
    d_grad = 0.0;
    d_haveval=false;
  }
  
  T getGrad() 
  {
    return d_grad;
  }

  // this function is key to the magic
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
      // getName removed
      if(doLog) std::cout<<"I'm a param called '"<<""<<"' with value "<<d_val<<", nothing to backward further"<<std::endl;

      if(doLog) std::cout<<"   receiving a d_grad of "<< (d_grad) <<std::endl;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<""<<"="<<d_val<<"\\ng="<<d_grad<<"\"]\n";;
      return;
    }
    else if(d_mode == Modes::Addition) {
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n+\"]\n";;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      if(doLog) std::cout<<"Addition going left, delivering "<< (d_grad)<<std::endl;

      d_lhs->d_grad += d_grad;
      if(doLog) std::cout<<"Addition going right, delivering "<< (d_grad) <<std::endl;

      d_rhs->d_grad += d_grad;
    }
    else if(d_mode == Modes::Mult) {
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<d_val<<"\\n*\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_rhs.get()<<"\"\n";

      if(doLog) std::cout<<"Mult lhs grad: \n"<<d_lhs->d_grad<<std::endl;
      T tposed = d_rhs->d_val; // this used to say "transpose"

      if(doLog) std::cout<<"Our grad:\n";
      if(doLog) std::cout<<d_grad<<std::endl;

      if(doLog) std::cout<<"Mult rhs grad: \n"<<d_rhs->d_grad<<std::endl;
      
      T tposed2 = d_lhs->d_val; // this used to say "transpose"
      
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
      T tposed = d_rhs->d_val; // this used to say transpose

      if(doLog) std::cout<<"Our grad:\n";
      if(doLog) std::cout<<d_grad<<std::endl;

      if(doLog) std::cout<<"Div rhs grad: \n"<<d_rhs->d_grad<<std::endl;
      
      T tposed2 = d_lhs->d_val; // this used to say transpose
      
      if(doLog) std::cout<<"Going to left, delivering "<< ( d_grad/tposed )<<std::endl;

      d_lhs->d_grad+=(d_grad/tposed);
      if(doLog) std::cout<<"Going right, delivering " << (-d_grad*tposed2/(tposed*tposed) ) <<std::endl;

      d_rhs->d_grad+=(-d_grad*tposed2/(tposed*tposed));
    }
    else if(d_mode == Modes::Func) {
      if(doLog) std::cout<<"Function, delivering "<< (d_grad  * getDeriv<T>(g_fss[d_findex])(d_lhs->d_val)) << std::endl;
      if(doLog) std::cout<<"Function, our grad "<<d_grad << std::endl;
      if(doLog) std::cout<<"Function, our val "<<d_val << std::endl;
      if(doLog) std::cout<<"Function, our val "<<d_findex<< std::endl;
      
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<"func\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";

      d_lhs->d_grad+=(d_grad*getDeriv<T>(g_fss[d_findex])(d_lhs->d_val));
    }
    else if(d_mode == Modes::Max) {
      if(doLog) std::cout<<"Max... our grad " << d_grad <<std::endl;
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" [label=\""<<"max\"]\n";
      if(g_tree) g_tree<<'"'<<(void*)this<< "\" -> \""<<(void*)d_lhs.get()<<"\"\n";

      auto delta = d_lhs->d_val < d_rhs->d_val; // magic vector
      d_rhs->d_grad += delta * d_grad;
      
      delta = !delta;
      d_lhs->d_grad += delta * d_grad;
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
  T d_grad = 0.0; // 4
  uint8_t d_findex = 0; // 1

  Modes d_mode{Modes::Unassigned};
  mutable bool d_haveval = false;
  mutable bool d_needsgrad = false;
  mutable bool d_variable = false;
  void needsGrad()
  {
    d_needsgrad=true;
  }
  void setVariable()
  {
    d_variable = true;
  }
};

template<typename Q>
struct WorkItem
{
  Q ourval=0.0; // 4
  unsigned int lhs=0, rhs=0; // 8
  
  uint8_t findex; // 1
  typename TrackedNumberImp<Q>::Modes mode; // 1
  bool needsgrad = 0;
};

template<typename T>
struct Work
{
  std::vector<WorkItem<T>> work;
  std::vector<T> grads;
  std::vector<std::pair<unsigned int, TrackedNumberImp<float>*>> dyns; // XXX this is not really generic!

  template<typename Q>
  Work<Q> convert()
  {
    Work<Q> ret;
    ret.work.reserve(work.size());
    ret.grads.reserve(grads.size());
    for(auto& g : grads) {
      ret.grads.push_back(Q(g));
    }
    
    for(auto& w : work) {
      WorkItem<Q> wi;
      wi.ourval = w.ourval;
      wi.lhs = w.lhs;
      wi.rhs = w.rhs;
      wi.findex = w.findex;
      wi.mode = (typename TrackedNumberImp<Q>::Modes)w.mode;
      wi.needsgrad = w.needsgrad;
      
      ret.work.push_back(wi);
    }
    return ret;
  }
  
  T getResult()
  {
    for(auto& w : work) {
      if(w.mode == TrackedNumberImp<T>::Modes::Parameter) {
      }
      else if(w.mode == TrackedNumberImp<T>::Modes::Addition) {
        w.ourval = work[w.lhs].ourval + work[w.rhs].ourval;
        w.needsgrad = work[w.lhs].needsgrad | work[w.rhs].needsgrad;
      }
      else if(w.mode == TrackedNumberImp<T>::Modes::Mult) {
        w.ourval = work[w.lhs].ourval * work[w.rhs].ourval;
        w.needsgrad = work[w.lhs].needsgrad | work[w.rhs].needsgrad;
      }
      else if(w.mode == TrackedNumberImp<T>::Modes::Div) {
        w.ourval = work[w.lhs].ourval / work[w.rhs].ourval;
        w.needsgrad = work[w.lhs].needsgrad | work[w.rhs].needsgrad;
      }
      else if(w.mode == TrackedNumberImp<T>::Modes::Max) {
        w.ourval = maxFunc(work[w.lhs].ourval, work[w.rhs].ourval);
        w.needsgrad = work[w.lhs].needsgrad | work[w.rhs].needsgrad;
      }
      else if(w.mode == TrackedNumberImp<T>::Modes::Func) {
        w.ourval = getFunc<T>(g_fss[w.findex])(work[w.lhs].ourval);
        w.needsgrad = work[w.lhs].needsgrad | work[w.rhs].needsgrad;
      }
    }
    return work.rbegin()->ourval;
  }
  // typically touches almost everything
  void zeroGrad()
  {
    memset((void*)&grads[0], 0, grads.size() * sizeof(grads[0]));
  }
  
  //! import possibly changed variable items
  void syncVariable() 
  {
    for(auto& g : dyns) {
      //   cout<<"Setting variable ["<<g.first<<"] to "<<g.second->d_val<<"\n";
      // note that this is not productive on non-parameter values
      work[g.first].ourval = g.second->d_val;
    }
  }
  
  //! export possibly changed variable items
  void syncBack() 
  {
    for(auto& g : dyns) {
      //   cout<<"Setting variable ["<<g.first<<"] to "<<g.second->d_val<<"\n";
      // note that when syncing back, we override the Mode to simple parameter
      g.second->d_lhs.reset();
      g.second->d_rhs.reset();
      g.second->d_mode = TrackedNumberImp<T>::Modes::Parameter;
      g.second->d_val = work[g.first].ourval;
      g.second->d_haveval = true; 
    }
  }
  
  
  //! export gradients
  void syncGrad()
  {
    for(auto& g : dyns) {
      if(g.second->d_needsgrad)
        g.second->d_grad = grads[g.first];
    }
  }
  //! export gradients additively
  void syncAddGrad()
  {
    for(auto& g : dyns) {
      if(g.second->d_needsgrad)
        g.second->d_grad += grads[g.first];
    }
  }
  
  
  void backward()
  {
    *grads.rbegin() = 1.0;
    for(int pos = work.size() -1 ; pos >= 0 ; --pos) {
      auto iter = &work[pos];
      if(iter->mode == TrackedNumberImp<T>::Modes::Parameter) {
        // do nothing
      }
      else if(iter->mode == TrackedNumberImp<T>::Modes::Addition) {
        grads[iter->lhs] += grads[pos];
        grads[iter->rhs] += grads[pos];
      }
      else if(iter->mode == TrackedNumberImp<T>::Modes::Mult) {
        grads[iter->lhs] +=(grads[pos] * work[iter->rhs].ourval);
        grads[iter->rhs] +=(work[iter->lhs].ourval * grads[pos]);
      }
      else if(iter->mode == TrackedNumberImp<T>::Modes::Div) {
        grads[iter->lhs] +=(grads[pos]/work[iter->rhs].ourval);
        grads[iter->rhs] +=(-grads[pos] * work[iter->rhs].ourval / (work[iter->rhs].ourval * work[iter->rhs].ourval));
      }
      else if(iter->mode == TrackedNumberImp<T>::Modes::Func) {
        grads[iter->lhs] +=(grads[pos] * getDeriv<T>(g_fss[iter->findex])(work[iter->lhs].ourval));
      }
      else if(iter->mode == TrackedNumberImp<T>::Modes::Max) {
        auto delta = work[iter->lhs].ourval < work[iter->rhs].ourval;
        grads[iter->rhs] += delta * grads[pos];
        
        delta = !delta;
        grads[iter->lhs]+= delta * grads[pos];
      }
      else
        abort();
    }
  }
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

  void needsGrad()
  {
    impl->needsGrad();
  }

  void setVariable()
  {
    impl->setVariable();
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

  std::vector<TrackedNumberImp<T>* > getTopo()
  {
    std::vector<TrackedNumberImp<T>* > topo;
    std::unordered_set<TrackedNumberImp<T>* > visited;
    impl->build_topo(visited, topo);
    topo.shrink_to_fit();
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

  

  template<typename Q>
  Work<Q> getWork(std::vector<TrackedNumberImp<T>* >& topo)
  {
    std::unordered_map<TrackedNumberImp<T>*, unsigned int> nums;

    unsigned int num=0;
    Work<Q> w;
    // note that this includes TrackedNumbers that start out as operations!
    for(auto iter = topo.begin(); iter != topo.end(); ++iter) {
      if((*iter)->d_needsgrad || (*iter)->d_variable)
        w.dyns.emplace_back(num, *iter);
      nums[*iter] = num++;
    }
    num--;

    for(auto iter = topo.begin(); iter != topo.end(); ++iter) {
      WorkItem<Q> wi;
      auto& item = **iter;
      wi.mode = (typename TrackedNumberImp<Q>::Modes) item.d_mode;
      if(item.d_lhs)
        wi.lhs = nums[item.d_lhs.get()];
      if(item.d_rhs)
        wi.rhs = nums[item.d_rhs.get()];
      if(item.d_mode == TrackedNumberImp<T>::Modes::Func) {
        wi.findex = item.d_findex;
      }
      if(item.d_mode == TrackedNumberImp<T>::Modes::Parameter)
        wi.ourval = item.d_val;
      wi.needsgrad = item.d_needsgrad;
      w.work.push_back(wi);
    }
    w.work.shrink_to_fit();
    w.grads.resize(w.work.size());
    return w;
  }

  std::shared_ptr<TrackedNumberImp<T>> impl;
};

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const WorkItem<T>& w) 
{
  static std::vector<std::string> modes={"unk", "=", "+", "*", "/", "f()", "max"}; // XXX WART 
  out<< w.ourval<<" "<<modes.at((int)w.mode)<<": ["<<w.lhs<<"], ["<<w.rhs<<"] needsgrad " <<w.needsgrad;
  return out;
}

template<typename T>
TrackedNumber<T> operator+(const TrackedNumber<T>& lhs, const TrackedNumber<T>& rhs)
{
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Addition;
  ret.impl->d_lhs = lhs.impl;
  ret.impl->d_rhs = rhs.impl;
  ret.impl->d_grad = 0.0;
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
  ret.impl->d_grad = 0.0;
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

template<typename T>
TrackedNumber<T> makeFunc(const TrackedNumber<T>& lhs, uint8_t findex)
{
  TrackedNumber<T> ret;
  ret.impl = std::make_shared<TrackedNumberImp<T>>();
  ret.impl->d_mode = TrackedNumberImp<T>::Modes::Func;
  ret.impl->d_findex = findex;
  ret.impl->d_lhs = lhs.impl;
  
  return ret;
}

typedef TrackedNumber<float> TrackedFloat;

