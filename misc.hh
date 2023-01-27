#pragma once
#include <iostream>
#include <random>
#include <algorithm>
#include <deque>
#include <vector>
#include <chrono>
#include <mutex>
#include <memory>
#include <atomic>
#include <optional>

struct HyperParameters
{
  float lr;
  float momentum;
  int batchMult;
  unsigned int getBatchSize()
  {
    return 8*batchMult;
  }
};

struct TrainingProgress
{
  int batchno=0;
  float lastTook=0;
  std::vector<float> losses;
  std::vector<float> corrects;
  std::atomic<unsigned int> trained=0;
};

extern struct TrainingProgress g_progress;
extern std::shared_ptr<HyperParameters> g_hyper;
int graphicsThread();

class Batcher
{
public:
  explicit Batcher(int n, std::optional<std::mt19937> rng=std::optional<std::mt19937>())
  {
    for(int i=0; i < n ; ++i)
      d_store.push_back(i);

    randomize(rng);
  }

  explicit Batcher(const std::vector<int>& in)
  {
    for(const auto& i : in)
      d_store.push_back(i);
    randomize();
  }

  auto getBatch(int n)
  {
    std::deque<int> ret;
    for(int i = 0 ; !d_store.empty() && i < n; ++i) {
      ret.push_back(d_store.front());
      d_store.pop_front();
    }
    return ret;
  }

  auto getBatchLocked(int n)
  {
    std::deque<int> ret;
    std::lock_guard<std::mutex> l(d_mut);
    for(int i = 0 ; !d_store.empty() && i < n; ++i) {
      ret.push_back(d_store.front());
      d_store.pop_front();
    }
    return ret;
  }

private:
  std::deque<int> d_store;
  std::mutex d_mut;
  void randomize(std::optional<std::mt19937> rnd = std::optional<std::mt19937>())
  {
    if(rnd) {
      std::shuffle(d_store.begin(), d_store.end(), *rnd);
    }
    else {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(d_store.begin(), d_store.end(), g);
    }
  }

};


struct DTime
{
  void start()
  {
    d_start =   std::chrono::steady_clock::now();
  }
  uint32_t lapUsec()
  {
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()- d_start).count();
    start();
    return usec;
  }

  std::chrono::time_point<std::chrono::steady_clock> d_start;
};
