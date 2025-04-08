#!/bin/sh
cd ..
echo "Atualizando Sistema..."
sudo apt-get update

#### Install Argos Kilobot Plugin ####
echo "Baixando Kilobot Plugin..."
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1scehQwIHhLUSuWJBzcoJhfBUJIcDWOb8' -O Argos3-Kilobot
unzip Argos3-Kilobot

echo "Instalando Argos Kilobot Plugin..."
mkdir argos3-kilobot-master/build
cd argos3-kilobot-master/build
cmake -DCMAKE_BUILD_TYPE=Release ../src
make
sudo make install