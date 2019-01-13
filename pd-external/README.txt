Dependencies
----------

libtorch can be found at : https://pytorch.org
Choose C++, select your system, then download the zip file and unzip it. (See caveats if you're on macOS)

fftw can be downloaded with : wget http://www.fftw.org/fftw-3.3.8.tar.gz
You should compile fftw with single precision support and position independant code : unzip the tarball, cd to the fftw directory, then

./configure --enable-float --with-pic
make

libsndfile can be found here : http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz


Building
--------

On macOS and Linux, you should be able to run the build script and be done with it.
If you built the dependencies in another directory, you can pass the paths to the libraries in the variables FFTW_DIR and TORCH_DIR, eg.

export FFTW_DIR=/path/to/fftw
export TORCH_DIR=/path/to/libtorch
./build


Running
-------

The external is bundled with its dependencies in the vae_sampler directory. Add this directory to Pd's search path to use the external. You can also cd in vae_sampler and open the testbed.pd patch to check that the external can be loaded by Pd.

Caveats
-------

With the macOS version of libtorch, you have to manually add libiomp and libmkml to the libtorch/lib folder. These can be found here : https://github.com/intel/mkl-dnn/releases

The test patches use various externals that may not be present in your pd distribution (eg. pd vanilla). Install the cyclone library to get the full functionality.

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

Ubuntu 18.04.1 LTS
cmake 3.10.2
GNU Make 4.1
gcc 7.3.0
libtorch 1.0.0.dev20190103
libsndfile 1.0.28
fftw 3.3.8
Pd vanilla 0.48.1
