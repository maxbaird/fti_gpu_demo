# FTI GPU Demo
This project demonstrates how FTI can be used with CUDA. It performs a simple
vector addition of C = A + B by dividing the vector size evenly among the number
of MPI processes. Each process then launches a CUDA kernel to compute their
partition of the vector.

## Compiling
To compile the following environment variables need to be set:
+ MPI_HOME
+ CUDA_HOME
+ FTI_HOME

Otherwise the Makefile can be adjusted directly to point to these locations.

These variables should point to the home directory of MPI, CUDA and FTI
respectively. To compile run `make`.

## Running
Execute the binary with the following argument
1. vector-size

+ _vector-size_ Specifies the length of the vector

You will need to have FTI built and configured for a successful run. For more
information on FTI see their [github](https://github.com/leobago/fti) repository.
