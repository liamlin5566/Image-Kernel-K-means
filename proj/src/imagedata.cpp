#include "imagedata.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


void imagedata::img2inputdata()
{
   
    m_data.reserve(m_height * m_width);

    for (std::size_t i=0; i < m_height; i++)
    {
        for (std::size_t j=0; j < m_width; j++)
        {
            //std::size_t index = i * width + j;
            std::vector<double> value(5);
            value[0] = double(m_img.at<cv::Vec3b>(i, j)[0]) / 255.0;
            value[1] = double(m_img.at<cv::Vec3b>(i, j)[1]) / 255.0;
            value[2] = double(m_img.at<cv::Vec3b>(i, j)[2]) / 255.0;
            value[3] = double(i) / m_height;
            value[4] = double(j) / m_width;

            m_data.push_back(value);
        }
    }
} 