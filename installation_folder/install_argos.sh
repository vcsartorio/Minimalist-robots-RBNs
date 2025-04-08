#!/bin/sh
cd ..
echo "Atualizando Sistema..."
sudo apt-get update

#### Install Argos ####
INSTALL_PKGS='build-essential cmake libfreeimage-dev libfreeimageplus-dev qt5-default qttools5-dev freeglut3-dev libxi-dev libxmu-dev liblua5.3-dev lua5.3 doxygen graphviz libgraphviz-dev asciidoc avr-gcc avr-libc gnome-terminal gperf libgoogle-perftools-dev'
echo "Instalando Dependencias..."
for i in $INSTALL_PKGS; do
    sudo apt-get install -y $i
done

echo "Baixando Argos..."
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=17hjk-rMVuif2CQlcPkAtZrJbomjwS0An' -O Argos3
unzip Argos3

echo "Instalando Argos Simulator..."
mkdir argos3-master/build_simulator
cd argos3-master/build_simulator
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DARGOS_BUILD_FOR=simulator \
    -DARGOS_BUILD_NATIVE=ON \
    -DARGOS_THREADSAFE_LOG=ON \
    -DARGOS_DYNAMIC_LOADING=ON \
    -DARGOS_USE_DOUBLE=ON \
    -DARGOS_DOCUMENTATION=ON \
    -DARGOS_INSTALL_LDSOCONF=ON \
    ../src

make
make doc
sudo make install
