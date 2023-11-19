#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <vector>
#include <iostream>
#include <cmath> 
#include "imagedata.hpp"

class kmeans{
public:
    kmeans(int n_clusters,  double gamma_c=0.6, double gamma_s=0.4):
    _n_clusters{n_clusters},
    _gamma_c{gamma_c},
    _gamma_s{gamma_s}
    {};

    int _n_clusters;
    double _gamma_c;
    double _gamma_s;
    int _max_iter = 1;
    int _k_cluster =3;

    std::vector<int> _alpha;

    void fit(imagedata& input);
    void save_fig(imagedata& input);
    void init(imagedata& input);

private:
    std::vector<std::vector<double>> calculate_gram_matrix(std::vector<std::vector<double>>& input_vec); // NxN, pp
    std::vector<std::vector<double>> calculate_dist_k(std::vector<std::vector<double>>& gram_matrix, std::vector<double> &diag); // Nxk, pp?
    std::vector<double> calculate_k_i_clsj(std::vector<std::vector<double>>& gram_matrix,  std::vector<int> &indices); // Nxk, pp?

    double calculate_square(std::vector<std::vector<double>>& gram_matrix, std::vector<int> &indices); // pp
    
    
};



std::vector<int> get_spec_index(std::vector<int> &vec, int cls); // Nxk. change to template, pp

std::vector<double> get_diag_from_Matrix(std::vector<std::vector<double>> &matrix); // NxN->Nx1. change to template, pp

std::vector<int> argmin(std::vector<std::vector<double>> &vec); // Nxk. change to template, pp

#endif