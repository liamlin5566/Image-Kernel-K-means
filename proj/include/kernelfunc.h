#ifndef KMEANS_HPP
#define KERNELFUNC_H

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void calculate_dist_k_cuda(const double *input, const int* alpha, double*dist_matrix, double mean_square, int N, int K, 
                                        double gamma_c, double gamma_s, int cls); // input_data: N*5, dist_matrix: N
__global__ void argmin_cuda(const double*dist_matrix, int *alpha, int N, int k);
__device__ double calculate_dist_cuda(const double * vec_i, const double * vec_j, const double gamma_c, const double gamma_s);


#endif