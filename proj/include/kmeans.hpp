#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <cstdlib> 
#include <vector>
#include <iostream>
#include <cmath> 
#include <omp.h>
#include <stdexcept>
#include "imagedata.hpp"

class kmeans{
public:
    kmeans(int n_clusters, double gamma_c=0.6, double gamma_s=0.4, int max_iter=1, double thresh=0.0001, int nthreads=4):
    m_clusters{n_clusters},
    m_gamma_c{gamma_c},
    m_gamma_s{gamma_s},
    m_max_iter{max_iter},
    m_thresh{thresh},
    m_nthreads{nthreads},
    m_verbose{false}
    {};

    
    std::vector<int> _alpha;

    void fit(imagedata& input);
    void fit_cuda(imagedata& input);
    void save_fig(imagedata& input, std::string outpath);
    void init(imagedata& input);

    void set_verbose(bool verbose) {m_verbose = verbose;}

    int k_cluster() {return m_clusters;}
    double gamma_c() {return m_gamma_c;}
    double gamma_s() {return m_gamma_s;}
    int max_iter() {return m_max_iter;}
    int nthreads() {return m_nthreads;}
    double thresh() {return m_thresh;}

private:
    
    std::vector<std::vector<double>> calculate_dist_k_omp(imagedata& input); // Nxk, pp?
    
    double check_change_omp(std::vector<int>&  cur_alpha); // Nxk, pp?
    double calculate_dist(std::vector<double>& vec_i, std::vector<double>& vec_j); // Nxk, pp?

    int m_clusters;
    double m_gamma_c;
    double m_gamma_s;
    int m_max_iter;
    double m_thresh; //check for early stop
    int m_nthreads;
    bool m_verbose;
    

};

std::vector<int> get_spec_index(std::vector<int> &vec, int cls); // Nxk. change to template, pp
std::vector<int> argmin_omp(std::vector<std::vector<double>> &vec, int nthreads=4); // Nxk. change to template, pp

#endif