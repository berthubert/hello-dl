#include "tracked.hh"
using namespace std;

ofstream g_tree("./tree.part");

int main()
{
  g_tree<<"digraph D {"<<endl;
  TrackedFloat x(-1.0, "x");
  TrackedFloat y(2.0, "y");
  TrackedFloat one(1.0, "one");
  TrackedFloat fact(2.0, "fact");
  TrackedFloat res = fact * doFunc(x*x+y*y*y, SigmoidFunc());
  
  cout << res.getVal() << " ==? " <<   1-SigmoidFunc::func(9.0) << endl;
  res.backward();
  // -(2*x * deriv(9.0)) = 2*deriv(9)
  cout<<x.getGrad() << " ==? " <<  2*SigmoidFunc::deriv(9.0) << endl;

  cout << y.getGrad() << " ==? " << -3*4*SigmoidFunc::deriv(9.0) << endl;
  g_tree<<"}"<<endl;
}
