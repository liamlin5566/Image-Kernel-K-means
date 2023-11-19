#include "imagedata.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


void imagedata::img2inputdata()
{
    int height = _img.rows;
	int width = _img.cols;

    for (size_t i=0; i < height; i++)
    {
        for (size_t j=0; j < width; j++)
        {
            size_t index = i * width + j;
            std::vector<double> value(5);
            value[0] = double(_img.at<cv::Vec3b>(i, j)[0]) / 255.0;
            value[1] = double(_img.at<cv::Vec3b>(i, j)[1]) / 255.0;
            value[2] = double(_img.at<cv::Vec3b>(i, j)[2]) / 255.0;
            value[3] = double(i) / height;
            value[4] = double(j) / width;

            data.push_back(value);
        }
    }
} 