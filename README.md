# hello-dl
A from scratch introduction to modern machine learning. While tutorials using
the industry mainstay PyTorch abound, this tutorial starts with matrices.

The goal is to have minimal software that is still going to wow you by being
pretty clever. 

The software meanwhile may be minimal, but will showcase modern deep
learning techniques.

This project was inspired by [Georgi Gerganov](https://ggerganov.com/)'s
AWESOME [C++ implementation of OpenAI's Whisper speech/translation model](https://github.com/ggerganov/whisper.cpp).

# Mission statement

 * Really make you feel what deep learning is all about
   * Tools must do things you will be personally be impressed by
 * Not skimp over the details, how does it ACTUALLY work
 * Make reading the tool's source code a good learning experience
 * Use language and terms that are compatible with modern day usage
   * So you can also understand PyTorch tutorials
 * However, also not confuse you with weird historical words
 * Cover all the techniques that form the backbone of modern DL successes

Non-goals:

 * Turn you into a machine learning professional
 * Teach you PyTorch

The idea is that after you are done with the blog posts (yet to be written) and
the tools that it should all make sense to you. 

# Status
So far this implements a small but pretty nice autograd system.  Initial
experiments (like `first-relu`) look reasonably ok.

This software is meant to accompany a series of blog posts introducing deep
learning from the ground up. That series hasn't started yet as I am still
figuring out how this stuff works.

If you want to see something cool already, take a look at [testrunner.cc]
which already shows some of the autogradient stuff.

# Getting started
Checkout the repository, run `cmake .` and then `make`.
To actually do something [download the
EMNIST](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip) dataset
and unzip it. There's no need to gunzip the .gz files.

Next up, run `./tensor` or `./first-relu` and wonder what you are seeing.

# Data
https://www.nist.gov/itl/products-and-services/emnist-dataset

https://arxiv.org/pdf/1702.05373v1
http://yann.lecun.com/exdb/mnist/

# Inspiration
https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb

