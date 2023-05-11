# 267-Project-RSA

An implementation of Ring Self Attention (RSA) in both UPC++ and MPI.

To build the MPI version run the following within the RSA-MPI subdirectory:
```
source modules.sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release
```

To compile the UPC++ version run the following within the RSA-UPC subdirectory:
```
source modules.sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_CXX_COMPILER=/opt/cray/pe/craype/2.7.19/bin/CC
```
