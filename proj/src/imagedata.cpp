#include "imagedata.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

template class imagedata<float>;
template class imagedata<double>;

template<typename DType>
void imagedata<DType>::img2inputdata()
{
    m_data = std::vector<std::vector<DType>>(m_height * m_width, std::vector<DType>(5, 0)); 
    for (std::size_t i=0; i < m_height; i++)
    {
        for (std::size_t j=0; j < m_width; j++)
        {
            m_data[i * m_width + j][0] = DType(m_img.at<cv::Vec3b>(i, j)[0]) / 255.0;
            m_data[i * m_width + j][1] = DType(m_img.at<cv::Vec3b>(i, j)[1]) / 255.0;
            m_data[i * m_width + j][2] = DType(m_img.at<cv::Vec3b>(i, j)[2]) / 255.0;
            m_data[i * m_width + j][3] = DType(i) / m_height;
            m_data[i * m_width + j][4] = DType(j) / m_width;
            
        }
    }
} 