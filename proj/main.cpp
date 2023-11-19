#include "include/imagedata.hpp"
#include "include/kmeans.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


int main()
{
    std::cout << "hello world" << std::endl;
    cv::Mat img = cv::imread("./image1.png");

    imagedata imgdata = imagedata(img);

    // for (int i=0; i <imgdata.data.size(); i++)
    // {
    //     for (int j = 0; j < imgdata.data[i].size(); j++)
    //     {
    //         std::cout << imgdata.data[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    kmeans cluser = kmeans(3);
    cluser.fit(imgdata);
    cluser.save_fig(imgdata);
    std::cout << "finish" << std::endl;
    return 0;
}
