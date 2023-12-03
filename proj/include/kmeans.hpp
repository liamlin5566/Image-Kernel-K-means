#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <vector>
#include <iostream>
#include <cmath> 
#include <omp.h>
#include "imagedata.hpp"

class kmeans{
public:
    kmeans(int n_clusters, double gamma_c=0.6, double gamma_s=0.4, int max_iter=1, int nthreads=4):
    m_clusters{n_clusters},
    m_gamma_c{gamma_c},
    m_gamma_s{gamma_s},
    m_max_iter{max_iter},
    m_nthreads{nthreads}
    {};

    
    std::vector<int> _alpha;

    void fit(imagedata& input);
    void save_fig(imagedata& input, std::string outpath);
    void init(imagedata& input);

    int k_cluster() {return m_clusters;}
    double gamma_c() {return m_gamma_c;}
    double gamma_s() {return m_gamma_s;}
    int max_iter() {return m_max_iter;}
    int nthreads() {return m_nthreads;}

private:
    
    std::vector<std::vector<double>> calculate_dist_k_omp(imagedata& input); // Nxk, pp?
    
    double check_change_omp(std::vector<int>&  cur_alpha); // Nxk, pp?
    double calculate_dist(std::vector<double>& vec_i, std::vector<double>& vec_j); // Nxk, pp?

    int m_clusters;
    double m_gamma_c;
    double m_gamma_s;
    int m_max_iter;
    int m_nthreads;

    //std::vector<std::vector<double>> calculate_gram_matrix(std::vector<std::vector<double>>& input_vec); // NxN, pp
    //std::vector<std::vector<double>> calculate_dist_k(std::vector<std::vector<double>>& gram_matrix, std::vector<double> &diag); // Nxk, pp?
    //std::vector<double> calculate_k_i_clsj(std::vector<std::vector<double>>& gram_matrix,  std::vector<int> &indices); // Nxk, pp?
    //double calculate_square(std::vector<std::vector<double>>& gram_matrix, std::vector<int> &indices); // pp

};

std::vector<int> get_spec_index(std::vector<int> &vec, int cls); // Nxk. change to template, pp
std::vector<int> argmin_omp(std::vector<std::vector<double>> &vec, int nthreads=4); // Nxk. change to template, pp
//std::vector<double> get_diag_from_Matrix(std::vector<std::vector<double>> &matrix); // NxN->Nx1. change to template, pp
//std::vector<int> argmin(std::vector<std::vector<double>> &vec); // Nxk. change to template, pp

#endif