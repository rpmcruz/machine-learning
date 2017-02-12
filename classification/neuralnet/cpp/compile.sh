#!/bin/sh
cd cpp
g++ -g -Wall -c -fPIC ranknet.cpp -o ranknet.o
g++ -shared -Wl,-soname,libranknet.so -o libranknet.so ranknet.o -larmadillo
