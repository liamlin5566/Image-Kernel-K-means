#ifndef IMAGEDATA_HPP
#define IMAGEDATA_HPP
#include <opencv2/opencv.hpp>
#include <vector>

class imagedata{
public:
    imagedata(cv::Mat &img)
    {
        _img = img;
        img2inputdata();
    }
    cv::Mat _img;
    std::vector<std::vector<double>> data;

private:
    void img2inputdata();
}; 

#endif