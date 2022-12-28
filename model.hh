#pragma once
#include "layers.hh"
#include <sstream>

struct ModelState
{
  std::vector<LayerBase*> d_members;
  void save(std::ostream& out) const
  {
    for(const auto& mem : d_members)
      mem->save(out);
  }
  void save(std::string& out) const
  {
    std::ostringstream os;
    for(const auto& mem : d_members)
      mem->save(os);

    out=os.str();
  }
  
  void load(std::istream& in)
  {
    for(auto& mem : d_members)
      mem->load(in);
  }

  void load(std::string& in)
  {
    std::istringstream is(in);
    load(is);
  }

  void learn(float lr)
  {
    for(auto& mem : d_members)
      mem->learn(lr);
  }
  
  void zeroGrad()
  {
    for(auto& mem : d_members)
      mem->zeroGrad();
  }

  uint32_t size()
  {
    size_t ret = 0;
    for(auto& mem : d_members)
      ret += mem->size();
    return ret;
  }
  
};
