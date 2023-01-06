#pragma once
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
std::vector<std::string> textChopper(const char* fname, size_t siz, int mult=1);

class BiMapper
{
public:
  explicit BiMapper(const char* fname, int lim=-1);
  int c2i(char c) const
  {
    auto iter = d_c2i.find(c);
    if(iter == d_c2i.end()) {
      std::cout<<("Attempting to find unknown character with value '"+std::to_string((int)c)+"'")<<std::endl;
      return 0;
    }
    return iter->second;
  }
  char i2c(int i) const
  {
    auto iter = d_i2c.find(i);
    if(iter == d_i2c.end()) {
      std::cout<<("Attempting to find unknown integer "+std::to_string(i))<<std::endl;
      return '?';
    }
    return iter->second;
  }

private:
  std::unordered_map<int, char> d_c2i;
  std::unordered_map<char, int> d_i2c;
};


