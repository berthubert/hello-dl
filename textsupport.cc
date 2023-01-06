#include "textsupport.hh"
#include <iostream>
#include <array>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <random>

using namespace std;

vector<string> textChopper(const char* fname, size_t siz, int mult)
{
  ifstream ifs(fname);

  vector<char> buffer(1024000);
  string total;
  while(!ifs.eof()) {
    ifs.read(&buffer[0], buffer.size());
    total.append(&buffer[0], &buffer[ifs.gcount()]);
  }
  buffer.clear();
  unsigned int pieces = mult*total.size()/siz;
  vector<string> ret;
  ret.reserve(pieces);

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(0, total.size() - siz -1);
  
  for(unsigned int n = 0 ; n < pieces; ++n) {
    ret.push_back(total.substr(distrib(gen), siz));
    for(auto& c : *ret.rbegin())
      if(c=='\n' || c=='\t') c=' ';
        
  }
  
  return ret;
}


BiMapper::BiMapper(const char* fname, int lim)
{
  ifstream ifs(fname);
  std::array<unsigned char, 4096> a;
  std::unordered_map<int, int> popcount;
  while(!ifs.eof()) {
    ifs.read((char*)&a[0], a.size());
    for(const auto& c : a) {
      //cout<<c;
      if(c<127)
        popcount[c]++;
    }
  }
  vector<pair<int,int>> revcount;
  for(const auto& p : popcount)
    revcount.push_back(p);
  sort(revcount.begin(), revcount.end(), [](const auto& a, const auto& b) {
    return b.second < a.second;
  });

  if(lim >= 0 && revcount.size() > (unsigned int)lim)
    revcount.resize(lim);
  
  for(unsigned int n=0; n < revcount.size(); ++n) {
    d_c2i[revcount[n].first] = n;
    d_i2c[n]=revcount[n].first;
    //    cout<<(char)revcount[n].first <<" -> "<<n<<"\n";
  }
  cout<<"Assigned "<<d_c2i.size()<<" mappings"<<endl;
}

