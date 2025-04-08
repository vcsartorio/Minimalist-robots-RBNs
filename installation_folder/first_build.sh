#!/bin/sh
cd ..
mkdir build
cd build
cmake ../src
make

pip3 install -U pymoo
pip3 install lxml
pip3 install scipy
pip3 install numpy
pip3 install matplotlib
pip3 install pandas
pip3 install lxml
pip3 install sympy
pip3 install seaborn
pip3 install colour