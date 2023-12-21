#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <cstdlib> 
#include <vector>
#include <iostream>
#include <cmath> 
#include <omp.h>
#include <stdexcept>

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernelfunc.h"
#endif

#include "imagedata.hpp"

template<typename DType>
class kmeans{
public:
    kmeans(int n_clusters, DType gamma_c=0.6, DType gamma_s=0.4, int max_iter=1, DType thresh=0.0001, int nthreads=4):
    m_clusters{n_clusters},
    m_gamma_c{gamma_c},
    m_gamma_s{gamma_s},
    m_max_iter{max_iter},
    m_thresh{thresh},
    m_nthreads{nthreads},
    m_verbose{false}
    {};

    void fit(imagedata<DType>& input); // clustering function

#ifdef __NVCC__
    void fit_cuda(imagedata<DType>& input); // clustering function (cuda)
#endif

    void save_fig(imagedata<DType>& input, std::string outpath); // save segmented image
    void init(imagedata<DType>& input); // random initialization

    void set_verbose(bool verbose) {m_verbose = verbose;}

    int k_cluster() {return m_clusters;}
    DType gamma_c() {return m_gamma_c;}
    DType gamma_s() {return m_gamma_s;}
    int max_iter() {return m_max_iter;}
    int nthreads() {return m_nthreads;}
    DType thresh() {return m_thresh;}

    std::vector<int> alpha; // cluster assignment 

private:
    
    std::vector<std::vector<DType>> calculate_dist_k_omp(imagedata<DType>& input); // calculate the distance between pixel and centers
    
    DType calculate_change_omp(std::vector<int>&  cur_alpha); // calculate the change of assignment
    DType calculate_dist(std::vector<DType>& vec_i, std::vector<DType>& vec_j); // calculate the distance (kernel function)

    int m_clusters; // number of clusters (k)
    DType m_gamma_c;  // the weights for color
    DType m_gamma_s; // the weights for spatial 
    int m_max_iter; // max iteration
    DType m_thresh; //check for early stop
    int m_nthreads; // to determine whether the change of assignment is samll
    bool m_verbose;
    

};

std::vector<int> get_spec_index(std::vector<int> &vec, int cls); // choose the indice with specific assignment
template<typename DType> std::vector<int> argmin_omp(std::vector<std::vector<DType>> &vec, int nthreads=4); //argmin along 1 axis

#endif