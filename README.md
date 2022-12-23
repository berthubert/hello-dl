# hello-dl
A from scratch introduction to modern machine learning. Many tutorials exist
already of course, but this one aims to really explain what is going on.
Other documents start out from the (very impressive) PyTorch environment, or
they attempt to math it up from first principles. 

Trying to understand deep learning via PyTorch is like trying to learn
aerodynamics from flying Airbus A380. 

Meanwhile the pure maths approach ("see it is easy, it is just a Jacobian
matrix") is probably only suited to people who dream in derivatives.

The goal of this tutorial is to develop modern neural networks entirely from
scratch, but where we still end up with really impressive results.

To do so, this project contains some minimalist tooling.  The software may
be minimal, but will showcase modern deep learning techniques that should
wow you into believing that something very special is going on.

This project was inspired by [Georgi Gerganov](https://ggerganov.com/)'s
AWESOME [C++ implementation of OpenAI's Whisper speech/translation model](https://github.com/ggerganov/whisper.cpp).

# Mission statement

 * Really make you feel what deep learning is all about
   * Tools must do things you will be personally be impressed by
 * Not skimp over the details, how does it ACTUALLY work
 * Make reading the tool's source code a good learning experience
 * However, also not confuse you with weird historical words
 * Cover all the techniques that form the backbone of modern DL successes
 * Use language and terms that are compatible with modern day usage
   * So you can also understand PyTorch tutorials
 * Provide modules that mirror functionality in PyTorch

Non-goals:

 * Turn you into a machine learning professional
 * Teach you PyTorch directly

The idea is that after you are done with the blog posts (yet to be written) and
have worked with the tools that it should all make sense to you. And *then*
you can get to work with professional tooling and get to work.

# Status
So far this implements a small but pretty nice autograd system. In
`first-convo.cc` you can find a ~1100 line total computer program that
learns to recognize handwritten digits in 10 minutes (90% accuracy so far).

```bash
git clone https://github.com/berthubert/hello-dl.git
cd hello-dl
cmake .
make -j4
wget http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
unzip gzip.zip
./first-convo
```
This will require 10GB of RAM for now, which seems a bit too much. But the
result is nice:

```
Start!
Have 240000 images
Configuring network................................................................
Tying... done
Getting topology.. 
(some time passes)
Percent batch correct: 32.8125%
Average loss: 2.1219. Predicted: 8, actual: 8: We got it right!
Loss: 1.89751, -2.44847 -2.54296 -2.43445 -1.92324 -3.07031 -2.50595 -2.27749 -2.1481 -1.89751 -2.26366
          .******           
         *XXXXXXXXX.        
        .XXXXX**XXXX        
        XXXX.    XXX.       
        XXX*     *XX*       
       .XXX.     .XX*       
        XXX.     *XX.       
        XXX.     XXX        
        *XXX.   XXXX        
         *XXX. .XXX*        
         .XXXXXXXXX*        
           *XXXXXXX         
            XXXXXX.         
          .XXXXXXX.         
         .XXXXXXXXX         
        .XXX*   XXX*        
        *XXX    XXX*        
        XXX*    XXX*        
        XXX.    *XX*        
        XXX*    XXX*        
        *XXX*.*XXXX*        
        *XXXXXXXXXX.        
         .XXXXXXX.          
           *****            
```

This software is meant to accompany a series of blog posts introducing deep
learning from the ground up. That series hasn't started yet as I am still
figuring out how this stuff works.

If you want to see something cool already, take a look at
[testrunner.cc](./testrunner.cc)
which already shows some of the autogradient stuff.

# Getting started
Checkout the repository, run `cmake .` and then `make`.
To actually do something [download the
EMNIST](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip) dataset
and unzip it. There's no need to gunzip the .gz files.

Next up, run `./tensor` or `./first-convo` and wonder what you are seeing.

# Data
https://www.nist.gov/itl/products-and-services/emnist-dataset

https://arxiv.org/pdf/1702.05373v1
http://yann.lecun.com/exdb/mnist/

# Inspiration
https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb

