#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernelfunc.h"
#endif

#include "kmeans.hpp"

template class kmeans<float>;
template class kmeans<double>;

template<typename DType>
void kmeans<DType>::save_fig(imagedata<DType>& input, std::string outpath)
{   
    int height = input.height();
	int width = input.width();

    //std::cout << width << " " << height << std::endl;
    cv::Mat output_fig = cv::Mat(height, width, CV_32FC3, 0.0);
    int count = 0;

    for (int cls=0; cls < m_clusters; cls++)
    {
        std::vector<int> indices = get_spec_index(alpha, cls);
        int M = indices.size();
        //std::cout << cls << " " << M << std::endl;
        DType mean_value_b = 0.0;
        DType mean_value_g = 0.0;
        DType mean_value_r = 0.0;

        // Choose average rgb value to represent clustering center
        for (int i = 0; i < M; i++)
        {
            int idx = indices[i];
            DType b = input[idx][0];
            DType g = input[idx][1]; 
            DType r = input[idx][2];  

            mean_value_b += b;
            mean_value_g += g;
            mean_value_r += r;
        }

        count += M;
        mean_value_b /= M;
        mean_value_g /= M;
        mean_value_r /= M;
        
        // Write segmented result to output image
        for (int i = 0; i < M; i++)
        {
            int idx = indices[i];
           
            DType y = input[idx][3];  
            DType x = input[idx][4];  

            int img_y = int(y * height + 0.5);
            int img_x = int(x * width + 0.5 );

            output_fig.at<cv::Vec3f>(img_y, img_x)[0] = float(mean_value_b) * 255;
            output_fig.at<cv::Vec3f>(img_y, img_x)[1] = float(mean_value_g) * 255;
            output_fig.at<cv::Vec3f>(img_y, img_x)[2] = float(mean_value_r) * 255;
        }
    }
    
    //std::cout << count << std::endl;
    //output_fig.convertTo(output_fig, CV_8U3C);
    cv::imwrite(outpath, output_fig);
}

template<typename DType>
void kmeans<DType>::init(imagedata<DType>& input)
{
    int N = input.size();
    alpha = std::vector<int>(N, 0);

    // random initialize
    srand(0);
    for (int i=0; i < N; i++)
    {   
        alpha[i] = rand() % m_clusters;
    }
}

/*
The function do clustering
input: Nx5 array
output: clustering assignment
*/
template<typename DType>
void kmeans<DType>::fit(imagedata<DType>& input)
{
    
    init(input); // initialize
    //save_fig(input, "init.png");
    int iter = 0;
    while (iter < m_max_iter)
    {
        // For each pixe,  calculate the distance between pixel and centers
        std::vector<std::vector<DType>> dist_matrix;
        dist_matrix = calculate_dist_k_omp(input); // dist_matrix: Nxk

        // Choose the assignment (alpha) which minimize the cost
        std::vector<int> cur_alpha = argmin_omp<DType>(dist_matrix, m_nthreads);
        // calculate the difference of current assignment and previous assignment
        DType diff = calculate_change_omp(cur_alpha);

        if (m_verbose)
            std::cout << "iter: " << iter << ", diff: " << diff << std::endl;
        if (iter > 0 && diff < m_thresh)
        {   
            if (m_verbose)
                std::cout << "Early Stop" << std::endl;
            break;
        }

        alpha = cur_alpha;
        iter++;
    }

    if (m_verbose)
        std::cout << "finish clustering" << std::endl;
}


template<typename DType>
DType kmeans<DType>::calculate_change_omp(std::vector<int>&  cur_alpha)
{
    DType diff = 0.0;
    int N = alpha.size();
    #pragma omp parallel num_threads(m_nthreads)
    {
        int i, id, nthrds;
        id = omp_get_thread_num(); 
        nthrds= omp_get_num_threads(); 

        DType sum = 0.0;
        for (i = id; i < N; i = i + nthrds)
        {
            if (alpha[i] != cur_alpha[i])
            {
                sum += 1.0;
            }
        }
        // sum up the result from all threads (critical section)
        #pragma omp critical
        diff += sum;
    }

    diff = diff / N;

    return diff;
}


template<typename DType>
DType kmeans<DType>::calculate_dist(std::vector<DType>& vec_i, std::vector<DType>& vec_j)
{
    // gaussian kernel function
    DType spatial_dist = pow(vec_i[3] - vec_j[3], 2) + pow(vec_i[4] - vec_j[4], 2);      
    DType color_dist = pow(vec_i[0] - vec_j[0], 2) + pow(vec_i[1] - vec_j[1], 2) + pow(vec_i[2] - vec_j[2], 2);
    spatial_dist = exp(- m_gamma_s * spatial_dist);
    color_dist =  exp(- m_gamma_c * color_dist);
    return spatial_dist * color_dist;
}


/*
In this function, I calculate the distance to k cluster centers for each pixel
Input: Nx5 array
Output: Nxk array
*/
template<typename DType>
std::vector<std::vector<DType>> kmeans<DType>::calculate_dist_k_omp(imagedata<DType>& input) 
{   
    int N = input.size();
    
    std::vector<std::vector<DType>> dist(N, std::vector<DType>(m_clusters, 0));

    for (int cls=0; cls < m_clusters; cls++)
    {   
        // First choose the data/pixels belonging to cls
        std::vector<int> indices = get_spec_index(alpha, cls);
        if (indices.size() == 0)   {
            continue;
        }
        int M = indices.size();
        
        // Then calculate the square of centers (pre-compute)
        DType mean_square = 0.0; 
        #pragma omp parallel num_threads(m_nthreads)
        {
            DType sum = 0.0;
            int id, nthrds;
            id = omp_get_thread_num(); 
            nthrds= omp_get_num_threads(); 
            for (int i = id; i < M; i = i + nthrds)
            {
                int idx_i = indices[i];
                for (int j = 0; j < M; j++) //symmetric matrix can faster
                {
                    int idx_j = indices[j];
                    sum += calculate_dist(input[idx_i], input[idx_j]);
                }
            }

            // critical section (sum up the results from all threads)
            #pragma omp critical
            mean_square += sum;
        }
        mean_square = mean_square / (M*M);
        //std::cout << cls << " " << mean_square<<" " << M <<  std::endl;
        
        // For each pixe, calculate the distance between pixel and center
        // using kernel trick to get distance (differnet from traditional K-means due to non-linear mapping)
        #pragma omp parallel num_threads(m_nthreads)
        {
            int i, id, nthrds;
            id = omp_get_thread_num(); 
            nthrds= omp_get_num_threads(); 

            for (i = id; i < N; i = i + nthrds)
            {
                DType sum_i = 0.0;
                DType dist_i_j = 0.0;

                DType diag = calculate_dist(input[i], input[i]);
                for (int j = 0; j < M ; j++) //symmetric matrix can faster
                {
                    int idx_j = indices[j];
                    if (idx_j < N)  
                        sum_i += calculate_dist(input[i], input[idx_j]); //gram_matrix[i][idx_j];
                    else
                        throw std::out_of_range("Index out of range");
                }

                dist_i_j = sum_i * 2 / M; 
                dist[i][cls] = diag - dist_i_j + mean_square;
            }

        }

    }   

    return dist;
}


template<typename DType>
std::vector<int> argmin_omp(std::vector<std::vector<DType>> &vec, int nthreads) // Nxk. change to template, pp
{
    int N = vec.size();
    int K = vec[0].size();

    std::vector<int> min_indices(N, 0);

    #pragma omp parallel num_threads(nthreads)
    {
        int i, id, nthrds;
        id = omp_get_thread_num(); 
        nthrds= omp_get_num_threads(); 

        for (i = id; i < N; i = i + nthrds)
        {
            DType min_value = vec[i][0];
            int min_idx = 0;
            for (int j = 1; j <K; j++)
            {
                if (vec[i][j] < min_value)
                {
                    min_value = vec[i][j];
                    min_idx = j;
                }
            }
            min_indices[i] = min_idx;
        }
    }

    return min_indices;
}


std::vector<int> get_spec_index(std::vector<int> &vec, int cls)
{
    std::vector<int> indices;
    int N = vec.size();

    for (int i = 0; i < N; i++)
    {
        if (vec[i] == cls)
        {
            indices.push_back(i);
        }
    }

    return indices;
}

#ifdef __NVCC__
template<typename DType>
void kmeans<DType>::fit_cuda(imagedata<DType>& input)
{
    init(input);
    
    // Prepare data
    int N = input.size();
    DType * host_dist_matrix = (DType *)malloc(N * m_clusters * sizeof(DType));
    DType * host_input_data = (DType *)malloc(N * 5 * sizeof(DType));
    int * host_alpha = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N * m_clusters; i++)
    {
        host_dist_matrix[i] = 0.0;
    }

    for (int i = 0; i < N; i++)
    {
        host_input_data[i*5+0] = input[i][0];
        host_input_data[i*5+1] = input[i][1];
        host_input_data[i*5+2] = input[i][2];
        host_input_data[i*5+3] = input[i][3];
        host_input_data[i*5+4] = input[i][4];
    }

    // Allocation cuda mem and copy from host
    DType *device_dist_matrix;
    DType *device_input_data;
    int *device_alpha;

    cudaMalloc(&device_dist_matrix, N * m_clusters * sizeof(DType));
    cudaMalloc(&device_input_data, N * 5 * sizeof(DType));
    cudaMalloc(&device_alpha, N * sizeof(int));

    cudaMemcpy(device_dist_matrix, host_dist_matrix, N * m_clusters * sizeof(DType), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_data, host_input_data, N * 5 * sizeof(DType), cudaMemcpyHostToDevice);
    cudaMemcpy(device_alpha, alpha.data(), N * sizeof(int), cudaMemcpyHostToDevice); // copy the alpha from vector
    dim3 blockSize(32);
    dim3 numBlock((N + 32-1)/ 32);

    int iter = 0;
    while (iter < m_max_iter)
    {   
        
        for (int cls=0; cls < m_clusters; cls++)
        {

            // First calculate the square of centers (pre-compute)
            DType mean_square = 0.0;
            DType M = 0.0;
            #pragma omp parallel num_threads(m_nthreads)
            {
                DType thread_sum = 0.0, thread_M = 0.0;
                int id, nthrds;
                id = omp_get_thread_num(); 
                nthrds= omp_get_num_threads(); 
                for (int i = id; i < N; i = i + nthrds)
                {
                    if (alpha[i] == cls) // choose the data belonging to cls
                    {
                        for (int j = 0; j < N; j++) 
                        {
                            if (alpha[j] == cls) // choose the data belonging to cls
                            {       
                                thread_sum += calculate_dist(input[i], input[j]);
                            } 
                        }
                        thread_M += 1;
                    }   
                }

                // sum up the result from all threads (critical section)
                #pragma omp critical
                {
                    mean_square += thread_sum; 
                    M += thread_M;
                }    
            }
            mean_square = mean_square / (M*M);
           
            // For each pixe,  calculate the distance between pixel and one specific (=cls) center
            if (M > 0)
            {
                calculate_dist_single_center_cuda<DType><<<numBlock, blockSize>>>(device_input_data, device_alpha, device_dist_matrix, mean_square, N, m_clusters, 
                                                             m_gamma_c, m_gamma_s, cls);
            }
                
        }
        cudaDeviceSynchronize();
        // Choose the assignment (alpha) which minimize the cost
        argmin_cuda<DType><<<numBlock, blockSize>>>(device_dist_matrix, device_alpha, N, m_clusters);
        cudaDeviceSynchronize();
        cudaMemcpy(host_alpha, device_alpha, N * sizeof(int), cudaMemcpyDeviceToHost);
        
        // calculate the difference of current assignment and previous assignment
        std::vector<int> cur_alpha = std::vector<int>(host_alpha, host_alpha+N);  
        DType diff = calculate_change_omp(cur_alpha);  
        if (m_verbose)
            std::cout << "iter: " << iter << ", diff: " << diff << std::endl;
        if (iter > 0 && diff < m_thresh)
        {   
            if (m_verbose)
                std::cout << "Early Stop" << std::endl;
            break;
        }

        alpha = cur_alpha;
        iter++;
    }

    // Release Memory
    free(host_dist_matrix);
    free(host_input_data);
    free(host_alpha);

    cudaFree(device_alpha);
    cudaFree(device_dist_matrix);
    cudaFree(device_input_data);

    if (m_verbose)
        std::cout << "finish clustering" << std::endl;
}
#endif