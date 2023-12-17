#include <cuda.h>
#include <cuda_runtime.h>
#include "kernelfunc.h"

__global__ void calculate_dist_k_cuda(const double *input, const int* alpha, double*dist_matrix, double mean_square, int N, int K, 
                                        double gamma_c, double gamma_s, int cls) // input_data: N*5, dist_matrix: N
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID < N)
    {
        
        double diag = calculate_dist_cuda(&input[threadID*5], &input[threadID*5], gamma_c, gamma_s); //spatial_dist * color_dist;
        double sum_i = 0.0;
        double dist_i_j = 0.0;
        double M = 0;
        for (int j = 0; j < N; j++)
        {
            if (alpha[j] == cls) // choose the data belonging to cls
            {
                sum_i += calculate_dist_cuda(&input[threadID*5], &input[j*5], gamma_c, gamma_s);
                M += 1;
            }
        }

        if (M > 0)
        {
            dist_i_j = sum_i * 2 / M; 
            dist_matrix[threadID*K+cls] = diag - dist_i_j + mean_square;
        }
    }   
}

__global__ void argmin_cuda(const double*dist_matrix, int *alpha, int N, int k)
{
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    
    double min_value = dist_matrix[threadID * k + 0];
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




__device__ double calculate_dist_cuda(const double * vec_i, const double * vec_j, const double gamma_c, const double gamma_s)
{
    double spatial_dist = pow(vec_i[3] - vec_j[3], 2) + pow(vec_i[4] - vec_j[4], 2);      
    double color_dist = pow(vec_i[0] - vec_j[0], 2) + pow(vec_i[1] - vec_j[1], 2) + pow(vec_i[2] - vec_j[2], 2);
    spatial_dist = exp(-gamma_s * spatial_dist);
    color_dist =  exp(-gamma_c * color_dist);
    return spatial_dist * color_dist;
}

