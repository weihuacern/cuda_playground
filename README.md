# CUDA Playground

## NVIDIA
### OVERVIEW
Code optimization is an important role in the position you’re applying for and this coding example is designed to understand how you will tackle such a task. This code example is designed similarly to a deep learning framework, and as such contains a concept of host tensors and device tensors which are stored in CPU memory and GPU memory respectively. These tensors provide an easy and clear mechanism to work with data both on and off the GPU. Your objective is to optimize the GPU path of the 8 iterations of “op_and_normalize” and provide an accurate result to that which operates on the CPU. The input tensor size of 4096 x 1024 can be assumed fixed.

In other words: optimize the 8 sequential calls of
“device_tensor<2> op_and_normalize(device_tensor<2> input)”
and match the result within the accuracy specified to the 8 sequential calls of
“host_tensor<2> op_and_normalize(host_tensor<2> input)”

### INSTALLATION
If you are on a linux system with CUDA 10 installed at /usr/local/cuda you should be able to simply type make in the main directory. If you are on a windows system you will need to figure out your own build setup with the files in src.

By default Makefile is setup to look for CUDA at /usr/local/cuda.  If this is not where your CUDA is you should be able to simply modify the line in the Makefile that reads “CUDA ?= /usr/local/cuda” and modify it to point at your local CUDA installation.

By default the makefile generates GPU code for all architectures supported by CUDA 10. If you have a previous version you may not be able to compute for later GPU generations. For example if you have a CUDA version that doesn’t support Turing, you may need to remove the line: -gencode=arch=compute_75,code=sm_75. Compiling for all architectures may be slow and you are encouraged to trim this list to the GPU generation you are using.

### EVALUATION
You will be evaluated based on the following categories:
Accuracy compared to original implementation
Performance as measured by the provided timer
Coding practices (readability and conciseness)

### WHAT CAN YOU MODIFY
You are allowed to modify any of the code included except where marked otherwise in main.cu (while maintaining accuracy of the CPU reference implementation). You are responsible for making sure your program is correctly timing the amount of time it takes for the GPU computation as well as ensuring its accuracy. You must take in the data stored in dA (also copied initially in dOut) which is placed on the GPU and provide a matching GPU result to what is computed in hOut.

It is unlikely you will need to modify the following files:
tensor.[cu, cuh]
device_tensor.[cu, cuh]
host_tensor.[cu,cuh]
utils.cuh
gpu_timer.cuh

However, it may be worthwhile to use, modify, or reimplement the operations in:
device_patterns.cuh
ops.cuh

Or to write your own GPU functions directly in main.cu or in a separate file.

### NOMENCLATURE
Device and GPU are used interchangeably. They refer to an object or data located on the GPU DRAM which can be accessed in GPU kernels and referred to/manipulated with CUDA calls.

Host and CPU are used interchangeably. They refer to an object or data located on the CPU/main DRAM which can be accessed only by CPU/Host functions. These objects can not be directly used in GPU functions or GPU kernels.