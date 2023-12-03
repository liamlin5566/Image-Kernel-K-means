#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "kmeans.hpp"

using namespace std;


void kmeans::save_fig(imagedata& input, std::string outpath)
{   
    int height = input.height();
	int width = input.width();

    std::cout << width << " " << height << std::endl;
    cv::Mat output_fig = cv::Mat(height, width, CV_32FC3, 0.0);

    std::cout << input.size() << std::endl;
    int count = 0;

    for (int cls=0; cls < m_clusters; cls++)
    {
        std::vector<int> indices = get_spec_index(_alpha, cls);
        int M = indices.size();
        
        double mean_value_b = 0.0;
        double mean_value_g = 0.0;
        double mean_value_r = 0.0;

        for (int i = 0; i < M; i++)
        {
            int idx = indices[i];
            double b = input[idx][0];
            double g = input[idx][1]; 
            double r = input[idx][2];  

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
           
            double y = input[idx][3];  
            double x = input[idx][4];  

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
    cv::imwrite(outpath, output_fig);
    std::cout << "finish" << std::endl;

}

void kmeans::init(imagedata& input)
{
    int N = input.size();

    int segment = N / m_clusters;
    _alpha = std::vector<int>(N, 0);

    for (int i=0; i < N; i++)
    {   
        if (i / segment < m_clusters)
            _alpha[i] = i / segment;
        else
            _alpha[i] = m_clusters - 1;
    }
}

void kmeans::fit(imagedata& input)
{

    init(input);
    save_fig(input, "init.png");

    // std::cout << "init alpha" << std::endl;
    // for (int i = 0; i < _alpha.size(); i++)
    // {
    //     std::cout << _alpha[i] << std::endl;
    // }


    // std::vector<std::vector<double>> data_array = input.data;

    //std::vector<std::vector<double>> gram_matrix = calculate_gram_matrix(input.data);
    ///std::vector<double> flat_diag = get_diag_from_Matrix(gram_matrix);

    // for (int i = 0; i < 10; i++)
    // {
    //     for (int j = 0; j < 10; j++)
    //     {
    //         std::cout << gram_matrix[i][j] << " " ;
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < 10; i++)
    // {
        
    //      std::cout << flat_diag[i] << " " << std::endl;
    // }

    int iter = 0;
    while (iter < m_max_iter)
    {

        std::vector<std::vector<double>> dist_matrix;
        dist_matrix = calculate_dist_k_omp(input);
        std::vector<int> cur_alpha = argmin_omp(dist_matrix, m_nthreads);

        double diff = check_change_omp(cur_alpha);
        std::cout << diff  << std::endl;
        if (iter > 0 && diff < 0.001)
        {
            std::cout << "Early Stop" << std::endl;
            break;
        }

        _alpha = cur_alpha;
        iter++;
    }

    /*int c0 = 0;
    int c1 = 0;
    int c2 = 0;
    std::cout << "init alpha" << std::endl;
    for (int i = 0; i < _alpha.size(); i++)
    {
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
    std::cout << c0 << " " << c1 << " " << c2 << std::endl;*/
}


double kmeans::check_change_omp(std::vector<int>&  cur_alpha)
{
    double diff = 0.0;

    int N = _alpha.size();
    //int NUM_THREADS = 4;
    #pragma omp parallel num_threads(m_nthreads)
    {
        int i, id, nthrds;
        id = omp_get_thread_num(); 
        nthrds= omp_get_num_threads(); 

        double sum = 0.0;
        for (i = id; i < N; i = i + nthrds)
        {
            if (_alpha[i] != cur_alpha[i])
            {
                sum += 1.0;
            }
        }

        #pragma omp critical
        diff += sum;
    }

    diff = diff / N;

    return diff;
}


double kmeans::calculate_dist(std::vector<double>& vec_i, std::vector<double>& vec_j)
{
    double spatial_dist = pow(vec_i[3] - vec_j[3], 2) + pow(vec_i[4] - vec_j[4], 2);      
    double color_dist = pow(vec_i[0] - vec_j[0], 2) + pow(vec_i[1] - vec_j[1], 2) + pow(vec_i[2] - vec_j[2], 2);
    spatial_dist = exp(- m_gamma_s * spatial_dist);
    color_dist =  exp(- m_gamma_c * color_dist);
    return spatial_dist * color_dist;
}


std::vector<std::vector<double>> kmeans::calculate_dist_k_omp(imagedata& input) // Nxk
{   
    int N = input.size();
    
    std::vector<std::vector<double>> dist(N, std::vector<double>(m_clusters, 0));

    for (int cls=0; cls < m_clusters; cls++)
    {
        std::vector<int> indices = get_spec_index(_alpha, cls);
        
        if (indices.size() == 0)   {
            continue;
        }
        int M = indices.size();
        
        double mean_square = 0.0; //calculate_square(gram_matrix, indices);
        // calculate squares mean
        //int NUM_THREADS = 4;
        #pragma omp parallel num_threads(m_nthreads)
        {
            double sum = 0.0;
            int id, nthrds;
            id = omp_get_thread_num(); 
            nthrds= omp_get_num_threads(); 
            for (int i = id; i < M; i = i + nthrds)
            {
                int idx_i = indices[i];
                for (int j = 0; j < M; j++) //symmetric matrix can faster
                {
                    int idx_j = indices[j];
                    sum += calculate_dist(input[idx_i], input[idx_j]);//gram_matrix[idx_i][idx_j]; 
                }
            }

            #pragma omp critical
            mean_square += sum;
        }
        mean_square = mean_square / (M*M);
        
        #pragma omp parallel num_threads(m_nthreads)
        {
            int i, id, nthrds;
            id = omp_get_thread_num(); 
            nthrds= omp_get_num_threads(); 

            for (i = id; i < N; i = i + nthrds)
            {
                double sum_i = 0.0;
                double dist_i_j = 0.0;

                double diag = calculate_dist(input[i], input[i]);
                for (int j = 0; j < M ; j++) //symmetric matrix can faster
                {
                    int idx_j = indices[j];
                    if (idx_j < N)  
                        sum_i += calculate_dist(input[i], input[idx_j]); //gram_matrix[i][idx_j];
                    else
                        std::cout << "Error !!!!! idx_j > N" << std::endl;
                }

                dist_i_j = sum_i * 2 / M; 
                dist[i][cls] = diag - dist_i_j + mean_square;
            }

        }

    }   

    return dist;
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


std::vector<int> argmin_omp(std::vector<std::vector<double>> &vec, int nthreads) // Nxk. change to template, pp
{
    int N = vec.size();
    int K = vec[0].size();

    std::vector<int> min_indices(N, 0);


   // int NUM_THREADS = 4;
    #pragma omp parallel num_threads(nthreads)
    {
        int i, id, nthrds;
        id = omp_get_thread_num(); 
        nthrds= omp_get_num_threads(); 

        for (i = id; i < N; i = i + nthrds)
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
            min_indices[i] = min_idx;
        }
    }

    return min_indices;
}



/*std::vector<std::vector<double>> kmeans::calculate_gram_matrix(std::vector<std::vector<double>> &input_vec)
{
    int N = input_vec.size();
    std::vector<std::vector<double>> gram_matrix(N, std::vector<double>(N, 0));
    
    for (int i=0; i < N; i++)
    {
        std::vector<double> vec_i = input_vec[i];
        for (int j=0; j < N; j++)
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
            
            dist[i][cls] = diag[i] - dist_i_j[i] + mean_square;
        }

    }   

    return dist;
}*/



/*std::vector<double> get_diag_from_Matrix(std::vector<std::vector<double>> &matrix)
{   
    
    int N = matrix.size();
    std::vector<double> flat_diag(N, 0);

    for (int i = 0; i < N; i++)
    {
        flat_diag[i] = matrix[i][i];
    }
    return flat_diag;
}*/


/*std::vector<int> argmin(std::vector<std::vector<double>> &vec) // Nxk. change to template, pp
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
}*/

