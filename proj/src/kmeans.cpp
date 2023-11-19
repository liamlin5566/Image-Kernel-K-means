#include "kmeans.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;


void kmeans::save_fig(imagedata& input)
{   
    int height = input._img.rows;
	int width = input._img.cols;

    cv::Mat output_fig = cv::Mat(height, width, CV_32FC3, 0.0);

    std::cout << input.data.size() << std::endl;
    int count = 0;

    for (int cls=0; cls < _k_cluster; cls++)
    {
        std::vector<int> indices = get_spec_index(_alpha, cls);
        int M = indices.size();
        
        double mean_value_b = 0.0;
        double mean_value_g = 0.0;
        double mean_value_r = 0.0;

        for (int i = 0; i < M; i++)
        {
            int idx = indices[i];
            double b = input.data[idx][0];
            double g = input.data[idx][1]; 
            double r = input.data[idx][2];  

            mean_value_b += b;
            mean_value_g += g;
            mean_value_r += r;
        }

        count += M;
        mean_value_b /= M;
        mean_value_g /= M;
        mean_value_r /= M;

        
        for (int i = 0; i < M; i++)
        {
            int idx = indices[i];
           
            double y = input.data[idx][3];  
            double x = input.data[idx][4];  

            int img_y = int(y * height + 0.5);
            int img_x = int(x * width + 0.5 );

            //std::cout << img_y << " " << img_x << std::endl; 
            output_fig.at<cv::Vec3f>(img_y, img_x)[0] = float(mean_value_b) * 255;
            output_fig.at<cv::Vec3f>(img_y, img_x)[1] = float(mean_value_g) * 255;
            output_fig.at<cv::Vec3f>(img_y, img_x)[2] = float(mean_value_r) * 255;
        }
    }
    
    std::cout << count << std::endl;
    //output_fig.convertTo(output_fig, CV_8U3C);
    cv::imwrite("A.png", output_fig);
    std::cout << "finish" << std::endl;

}

void kmeans::init(imagedata& input)
{
    int N = input.data.size();

    int segment = N / _k_cluster;
    int conut = 0;

    _alpha = std::vector<int>(N, 0);

    
    for (int i=0; i < N; i++)
    {   
        if (i / segment < _k_cluster)
            _alpha[i] = i / segment;
        else
            _alpha[i] = _k_cluster - 1;
    }
}

void kmeans::fit(imagedata& input)
{

    init(input);

    // std::cout << "init alpha" << std::endl;
    // for (int i = 0; i < _alpha.size(); i++)
    // {
    //     std::cout << _alpha[i] << std::endl;
    // }


    // std::vector<std::vector<double>> data_array = input.data;

    std::vector<std::vector<double>> gram_matrix = calculate_gram_matrix(input.data);
    std::vector<double> flat_diag = get_diag_from_Matrix(gram_matrix);

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            std::cout << gram_matrix[i][j] << " " ;
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < 10; i++)
    {
        
         std::cout << flat_diag[i] << " " << std::endl;
    }

    int iter = 0;
    while (iter < _max_iter)
    {

        std::vector<std::vector<double>> dist_matrix;
        dist_matrix = calculate_dist_k(gram_matrix, flat_diag);

        _alpha = argmin(dist_matrix);

        iter++;
    }

    int c0 = 0;
    int c1 = 0;
    int c2 = 0;

    std::cout << "init alpha" << std::endl;
    for (int i = 0; i < _alpha.size(); i++)
    {
        //std::cout << _alpha[i] << std::endl;
        if      (_alpha[i] == 0) 
        {  
            c0 += 1;
        }
        else if (_alpha[i] == 1) 
        {
            c1 += 1;
        }
        else {c2 += 1;}
    }

    std::cout << c0 << " " << c1 << " " << c2 << std::endl;
}


std::vector<std::vector<double>> kmeans::calculate_gram_matrix(std::vector<std::vector<double>> &input_vec)
{
    int N = input_vec.size();
    std::vector<std::vector<double>> gram_matrix(N, std::vector<double>(N, 0));
    
    for (int i=0; i < input_vec.size(); i++)
    {
        std::vector<double> vec_i = input_vec[i];
        for (int j=0; j <input_vec.size(); j++)
        {
            std::vector<double> vec_j = input_vec[j];
            double spatial_dist = pow(vec_i[3] - vec_j[3], 2) + pow(vec_i[4] - vec_j[4], 2);
            
            double color_dist = pow(vec_i[0] - vec_j[0], 2) + pow(vec_i[1] - vec_j[1], 2) + pow(vec_i[2] - vec_j[2], 2);

            spatial_dist = exp(- _gamma_s * spatial_dist);
            color_dist =  exp(- _gamma_c * color_dist);

            gram_matrix[i][j] = spatial_dist * color_dist;
        }   
    }

    return gram_matrix;
}

std::vector<std::vector<double>> kmeans::calculate_dist_k(std::vector<std::vector<double>> &gram_matrix, std::vector<double> &diag) // Nxk
{   
    int N = gram_matrix.size();
    
    std::vector<std::vector<double>> dist(N, std::vector<double>(_k_cluster, 0));

    for (int cls=0; cls < _k_cluster; cls++)
    {
        std::vector<int> indices = get_spec_index(_alpha, cls);
        
        if (indices.size() == 0)   {
            continue;
        }
        

        double mean_square = calculate_square(gram_matrix, indices);
        std::vector<double> dist_i_j = calculate_k_i_clsj(gram_matrix, indices);

        for (int i=0; i < N; i++)
        {
            //  if (i < 200)
            // {
            //     std::cout << "dist: " <<diag[i] << " " << dist_i_j[i] << " " << mean_square << std::endl;
            // }
            
            dist[i][cls] = diag[i] - dist_i_j[i] + mean_square;
        }

    }   

    return dist;
}



double kmeans::calculate_square(std::vector<std::vector<double>> &gram_matrix, std::vector<int> &indices) // pp
{
    double sum = 0;
    int N = gram_matrix.size();
    int M = indices.size();

    for (int i = 0; i < indices.size(); i++)
    {
        int idx_i = indices[i];
        for (int j = 0; j < indices.size(); j++) //symmetric matrix can faster
        {
            int idx_j = indices[j];

            if (idx_i < N && idx_j < N)  
                sum += gram_matrix[idx_i][idx_j];
            else
                std::cout << "Error !!!!! idx_i > N or idx_j > N" << std::endl;
        }
    }
    sum = sum / double(M * M);
    return sum;
}


std::vector<double> kmeans::calculate_k_i_clsj(std::vector<std::vector<double>>& gram_matrix,  std::vector<int> &indices)
{
    int N = gram_matrix.size();
    std::vector<double> dist_i_j(N, 0);
    int M = indices.size();

    for (int i = 0; i < N; i++)
    {   
        double sum_i = 0.0;
        for (int j = 0; j < M ; j++) //symmetric matrix can faster
        {
            int idx_j = indices[j];

             if (idx_j < N)  
                sum_i += gram_matrix[i][idx_j];
            else
                std::cout << "Error !!!!! idx_j > N" << std::endl;
        }

        dist_i_j[i] = sum_i * 2 / M; 
    }
    return dist_i_j;
}


std::vector<int> get_spec_index(std::vector<int> &vec, int cls)
{
    std::vector<int> indices;

    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i] == cls)
        {
            indices.push_back(i);
        }
    }

    return indices;
}


std::vector<double> get_diag_from_Matrix(std::vector<std::vector<double>> &matrix)
{   
    
    int N = matrix.size();
    std::vector<double> flat_diag(N, 0);

    for (int i = 0; i < N; i++)
    {
        flat_diag[i] = matrix[i][i];
    }
    return flat_diag;
}


std::vector<int> argmin(std::vector<std::vector<double>> &vec) // Nxk. change to template, pp
{
    int N = vec.size();
    int K = vec[0].size();

    std::vector<int> min_indices(N, 0);

    for (int i = 0; i < N; i++)
    {   
        double min_value = vec[i][0];
        int min_idx = 0;
        for (int j = 1; j <K; j++)
        {
            if (vec[i][j] < min_value)
            {
                min_value = vec[i][j];
                min_idx = j;
            }
        }

        if (i < 200)
        {
            std::cout <<vec[i][0]<< " " << vec[i][1] << " " <<vec[i][2] << std::endl;
            std::cout << min_idx << std::endl;
        }
        //
        min_indices[i] = min_idx;
    }
    return min_indices;
}
