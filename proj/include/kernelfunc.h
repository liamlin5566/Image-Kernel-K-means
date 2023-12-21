#ifndef KMEANS_HPP
#define KERNELFUNC_H

#include <cuda.h>
#include <cuda_runtime.h>

// For each pixe,  calculate the distance between pixel and one specific (=cls) center
template<typename DType>
__global__ void calculate_dist_single_center_cuda(const DType *input, const int* alpha, DType*dist_matrix, const DType mean_square, int N, int K, 
                                        const DType gamma_c, const DType gamma_s, int cls);
//argmin along 1 axis
template<typename DType>
__global__ void argmin_cuda(const DType*dist_matrix, int *alpha, int N, int k);


// calculate the distance (kernel function)
template<typename DType>
__device__ DType calculate_dist_cuda(const DType * vec_i, const DType * vec_j, const DType gamma_c, const DType gamma_s);

#endif