#pragma once
#include <deque>
#include <vector>

class Batcher
{
public:
  explicit Batcher(int n)
  {
    for(int i=0; i < n ; ++i)
      d_store.push_back(i);

    randomize();
  }

  explicit Batcher(const std::vector<int>& in)
  {
    for(const auto& i : in)
      d_store.push_back(i);
    randomize();
  }

  std::vector<int> getBatch(int n)
  {
    std::vector<int> ret;
    ret.reserve(n);
    for(int i = 0 ; !d_store.empty() && i < n; ++i) {
      ret.push_back(d_store.front());
      d_store.pop_front();
    }
    return ret;
  }
private:
  std::deque<int> d_store;
  void randomize()
  {
    std::random_device rd;
    std::mt19937 g(rd());
 
    std::shuffle(d_store.begin(), d_store.end(), g);
  }

};
