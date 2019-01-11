Dependency
----------

libtorch can be found at : https://pytorch.org
Choose C++, select your system, then download the zip file in the pd-external directory and unzip it. (See caveats if you're on macOS)

fftw can be downloaded with : wget http://www.fftw.org/fftw-3.3.8.tar.gz
You should compile fftw with single precision support : unzip the tarball, cd to the fftw directory, then

./configure --enable-float
make
make install

libsndfile is not mandatory, but useful to build the tests. It can be found here : http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz

Building
--------

You should be able to run the build script and be done with it.

Running
-------

Open the testbed.pd patch inside the vae_sampler directory.

Caveats
-------

With the macOS version of libtorch, you have to manually add libiomp and libmkml to the libtorch/lib folder. These can be found here : https://github.com/intel/mkl-dnn/releases

Currently, at least in the macOS version, you have to keep the external in the same folder as the libs directory containing the libtorch shared libraries. You can use the external from any patch provided you correctly set the external search paths in PureData's preferences.

Tested on
---------

macOS 10.13.2
cmake 3.12.4
GNU make 3.81
clang 900.0.39.2
libtorch macos 1.0.0
libsndfile 1.0.28
fftw 3.3.8
Pd vanilla 0.49.1

// Outdated ...
Ubuntu 18.04.1 LTS
cmake 3.10.2
GNU Make 4.1
gcc 7.3.0
libtorch 1.0.0.dev20190103
libsndfile 1.0.28
fftw 3.3.8
Pd vanilla 0.48.1
