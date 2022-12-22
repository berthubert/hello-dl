#include "tracked.hh"
#include <malloc.h>
using namespace std;

ofstream g_tree("./tree.part");

int main()
{
  
  g_tree<<"digraph D {"<<endl;

  TrackedFloat x(2.0);
  TrackedFloat y(3.0);
  TrackedFloat res = x + y * y; 
  cout << res.getVal() << " ==? 11"<< endl;
  res.backward();
  cout << x.getGrad() << " ==? 1 "<< endl; // 1
  cout << y.getGrad() << " ==? 6 "<<endl; // 2*y = 6
  
  g_tree<<"}"<<endl;
}


#if 0
  
  cout<<sizeof(TrackedNumberImp<float>)<<endl;
  cout<<sizeof(std::function<float(float)>)<<endl;
  cout<<sizeof(float(*)(float))<<endl;
  //  malloc_info(0, stdout);
  TrackedFloat a, b, c, d;
  //  malloc_info(0, stdout);
  malloc_stats();
  a=1;
  malloc_stats();
  b=1;
  malloc_stats();
  c=1;
  malloc_stats();
  d=1;
  malloc_stats();
  //  malloc_info(0, stdout);
  return 0;


#endif
