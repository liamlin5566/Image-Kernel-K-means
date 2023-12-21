#include "imagedata.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

template class imagedata<float>;
template class imagedata<double>;

template<typename T>
void imagedata<T>::img2inputdata()
{
    m_data = std::vector<std::vector<T>>(m_height * m_width, std::vector<T>(5, 0)); 
    for (std::size_t i=0; i < m_height; i++)
    {
        for (std::size_t j=0; j < m_width; j++)
        {
            m_data[i * m_width + j][0] = T(m_img.at<cv::Vec3b>(i, j)[0]) / 255.0;
            m_data[i * m_width + j][1] = T(m_img.at<cv::Vec3b>(i, j)[1]) / 255.0;
            m_data[i * m_width + j][2] = T(m_img.at<cv::Vec3b>(i, j)[2]) / 255.0;
            m_data[i * m_width + j][3] = T(i) / m_height;
            m_data[i * m_width + j][4] = T(j) / m_width;
            
        }
    }
} 