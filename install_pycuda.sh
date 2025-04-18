#!/bin/bash
#
# Reference for installing 'pycuda': https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu

set -e

if ! which nvcc > /dev/null; then
  echo "ERROR: nvcc not found"
  exit
fi

arch=$(uname -m)
folder=${HOME}/src
mkdir -p $folder

echo "** Install requirements"
sudo apt-get install -y build-essential python3-dev python3-pip
sudo apt-get install -y libboost-python-dev libboost-thread-dev
sudo -H pip3 install setuptools

# Find boost python library
BOOST_PYTHON_LIB=$(ls /usr/lib/$(arch)-linux-gnu/libboost_python*.so 2>/dev/null | head -1)
if [ -z "$BOOST_PYTHON_LIB" ]; then
  echo "ERROR: Cannot find libboost_python*.so"
  exit 1
fi
boost_pylib=$(basename $BOOST_PYTHON_LIB)
boost_pylibname=$(echo $boost_pylib | sed 's/\.so$//')
boost_pyname=$(echo $boost_pylibname | sed 's/^lib//')

echo "** Download pycuda-2019.1.2 sources"
OLD_DIR=$(pwd)
cd $folder
if [ ! -f pycuda-2019.1.2.tar.gz ]; then
  wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
fi

echo "** Build and install pycuda-2019.1.2"
CPU_CORES=$(nproc)
echo "** cpu cores available: " $CPU_CORES
tar xzvf pycuda-2019.1.2.tar.gz
cd pycuda-2019.1.2
python3 ./configure.py --python-exe=/usr/bin/python3 --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib/${arch}-linux-gnu --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib/${arch}-linux-gnu --boost-python-libname=${boost_pyname} --boost-thread-libname=boost_thread --no-use-shipped-boost
make -j$CPU_CORES
python3 setup.py build
sudo -H python3 setup.py install

cd $OLD_DIR

python3 -c "import pycuda; print('pycuda version:', pycuda.VERSION)"
