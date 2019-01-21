Brief
-----

A python wrapper around a C implementation of the Griffin-Lim algorithm.


Dependencies
----------

fftw can be downloaded with : wget http://www.fftw.org/fftw-3.3.8.tar.gz
You should compile fftw once with double precision and once with single precision support, and position independant code : unzip the tarball, cd to the fftw directory, then :

./configure --with-pic
make
./configure --enable-float --with-pic
make


Building
--------

On macOS and Linux, you should be able to run the build script and be done with it.
If you built fftw in another directory, you can pass the path in the variables FFTW_DIR

FFTW_DIR=/path/to/fftw ./build


Running
-------

See pyglim.test.py for an example of how to use the pyglim wrapper.

Tested on
---------

macOS 10.13.2
GNU make 3.81
clang 900.0.39.2
fftw 3.3.8
