#include "fvector.hh"
#include <iostream>
#include <stdlib.h>
#include "textsupport.hh"

using namespace std;

int main(int argc, char**argv)
{
  auto lines=textChopper("corpus.txt", 75, 10);
  cout<<"Got "<<lines.size()<<" lines"<<endl;
  for(const auto& l : lines)
    cout<<l<<"\n";
}
