Building
--------

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$(cd ../libtorch ; pwd) ..
make

Running
-------

Copy the external in the same folder as the libtorch directory and the testbed.pd patch, then open testbed.pd

Caveats
-------

To link against torch, we have to manually add libiomp and libmkml to the libtorch/lib folder. These can be found here : https://github.com/intel/mkl-dnn/releases

Currently, at least in the macOS version, you have to keep the external in the same folder as the libtorch package. You can use the external from any patch provided you correctly set the external search paths in PureData's preferences.

Tested on
---------

macOS 10.13.2
cmake 3.12.4
GNU make 3.81
clang 900.0.39.2
libtorch macos 1.0.0
Pd vanilla 0.49.1
