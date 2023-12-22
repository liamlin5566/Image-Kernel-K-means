#include <cuda.h>
#include <cuda_runtime.h>
#include "kernelfunc.h"

template __global__ void argmin_cuda<float>(const float*dist_matrix, int *alpha, int N, int k);
template __global__ void calculate_dist_single_center_cuda<float>(const float *input, const int* alpha, float*dist_matrix, const float mean_square, int N, int K, 
                                        const float gamma_c, const float gamma_s, int cls);

template __global__ void argmin_cuda<double>(const double*dist_matrix, int *alpha, int N, int k);
template __global__ void calculate_dist_single_center_cuda<double>(const  double *input, const int* alpha, double*dist_matrix, const  double mean_square, int N, int K, 
                                        const  double gamma_c, const  double gamma_s, int cls);

/*
In this, calculate the distance between pixel and one specific (=cls) center for each pixel
use kernel trick due to non-linear mapping
input: Nx5 -> (N*5,) 1D array
dist_matrix: NxK -> (N*K, ) 1D array
*/
template<typename DType>
__global__ void calculate_dist_single_center_cuda(const DType *input, const int* alpha, DType*dist_matrix, const DType mean_square, int N, int K, 
                                        const DType gamma_c, const DType gamma_s, int cls) 
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; // get x_i's index
    if (threadID < N)
    {
        // x_i ^2
        DType diag = calculate_dist_cuda<DType>(&input[threadID*5], &input[threadID*5], gamma_c, gamma_s); //spatial_dist * color_dist;
        DType sum_i = 0.0;
        DType dist_i_j = 0.0;
        DType M = 0;

        //calculate the similarity between x_i and X_j, where X_j is the pixel set belonging to cls
        for (int j = 0; j < N; j++)
        {
            if (alpha[j] == cls) // choose the data belonging to cls
            {
                sum_i += calculate_dist_cuda<DType>(&input[threadID*5], &input[j*5], gamma_c, gamma_s);
                M += 1;
            }
        }

        if (M > 0)
        {
            dist_i_j = sum_i * 2 / M; 
            dist_matrix[threadID*K+cls] = diag - dist_i_j + mean_square; // dist[i][cls] = diag - dist_i_j + mean_square;
        }
    }   
}

template<typename DType>
__global__ void argmin_cuda(const DType*dist_matrix, int *alpha, int N, int k)
{
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    
    DType min_value = dist_matrix[threadID * k + 0];
    int min_id = 0;
    if (threadID < N)
    {
        for (int i = 1; i < k; i++)
        {
            if (min_value > dist_matrix[threadID * k+i])
            {
                min_value = dist_matrix[threadID * k+i];
                min_id = i;
            }
        }
        alpha[threadID] = min_id;
    }
}


template<typename DType>
__device__ DType calculate_dist_cuda(const DType * vec_i, const DType * vec_j, const DType gamma_c, const DType gamma_s)
{
    DType spatial_dist = pow(vec_i[3] - vec_j[3], 2) + pow(vec_i[4] - vec_j[4], 2);      
    DType color_dist = pow(vec_i[0] - vec_j[0], 2) + pow(vec_i[1] - vec_j[1], 2) + pow(vec_i[2] - vec_j[2], 2);
    spatial_dist = exp(-gamma_s * spatial_dist);
    color_dist =  exp(-gamma_c * color_dist);
    return spatial_dist * color_dist;
}

